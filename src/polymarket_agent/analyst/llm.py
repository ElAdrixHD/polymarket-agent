"""Multi-provider LLM client for market analysis."""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from polymarket_agent.analyst.prompts import (
    QUERY_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_system_prompt,
)
from polymarket_agent.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory evaluation history: keeps the last N evaluations per market so
# the LLM can see how prices and its own assessments have evolved over time.
# Key = market condition_id (or question hash as fallback).
# ---------------------------------------------------------------------------
MAX_HISTORY = 5
_evaluation_history: dict[str, deque[dict[str, Any]]] = {}


class ParsedQuery(BaseModel):
    search_terms: list[str]
    max_hours: float | None = None
    action: str = "search"  # "search" or "analyze"
    risk_tolerance: str | None = None


class MarketAnalysis(BaseModel):
    recommendation: str  # BUY_YES, BUY_NO, SKIP
    confidence: float
    estimated_probability: float | None = 0.5
    reasoning: str
    suggested_price: float | None = None
    suggested_size: float | None = None

    def __init__(self, **data: Any) -> None:
        # Coerce null estimated_probability to default
        if data.get("estimated_probability") is None:
            data["estimated_probability"] = 0.5
        super().__init__(**data)


async def _call_openai(system: str, user: str) -> str:
    """Call LLM via OpenAI-compatible API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


async def _call_anthropic(system: str, user: str) -> str:
    """Call LLM via Anthropic SDK."""
    from anthropic import AsyncAnthropic

    base = settings.llm_base_url or None
    if base and base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/")[:-3]

    client = AsyncAnthropic(
        api_key=settings.llm_api_key,
        base_url=base,
    )
    resp = await client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Skip ThinkingBlocks (extended thinking) and grab the first TextBlock
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text
    return ""


async def chat(system: str, user: str) -> str:
    """Send a chat message to the configured LLM provider."""
    if settings.llm_provider == "anthropic":
        return await _call_anthropic(system, user)
    return await _call_openai(system, user)


def _parse_analysis(raw: str) -> MarketAnalysis:
    """Parse LLM JSON response, stripping markdown fences if present.

    If the JSON is truncated (e.g. cut-off reasoning string), attempts to
    repair it by closing open strings and braces.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to repair truncated JSON: close any open string and object
        repaired = text.rstrip()
        if repaired.endswith("\\"):
            repaired = repaired[:-1]
        # Close unclosed string
        quote_count = repaired.count('"') - repaired.count('\\"')
        if quote_count % 2 == 1:
            repaired += '"'
        # Close any trailing comma and open braces
        repaired = repaired.rstrip(",")
        open_braces = repaired.count("{") - repaired.count("}")
        repaired += "}" * open_braces
        data = json.loads(repaired)
    return MarketAnalysis(**data)


def _normalize_outcome(outcome: str) -> str:
    """Map market outcome labels to canonical YES/NO.

    Handles standard Yes/No markets and Up/Down markets.
    """
    key = outcome.strip().upper()
    if key in ("YES", "UP"):
        return "YES"
    if key in ("NO", "DOWN"):
        return "NO"
    return key


def extract_token_ids(market: dict[str, Any]) -> dict[str, str]:
    """Extract YES/NO token IDs from a Gamma market dict.

    Returns ``{"YES": "<token_id>", "NO": "<token_id>"}``.
    Handles both the ``tokens`` list format (events endpoint) and the
    ``clobTokenIds`` + ``outcomes`` JSON-string format (markets endpoint).
    Also maps Up/Down outcomes to YES/NO.
    """
    result: dict[str, str] = {"YES": "", "NO": ""}
    tokens = market.get("tokens") or []
    if tokens:
        for tok in tokens:
            key = _normalize_outcome(tok.get("outcome", ""))
            if key in result:
                result[key] = tok.get("token_id", "")
    else:
        clob_ids_raw = market.get("clobTokenIds", "[]")
        outcomes_raw = market.get("outcomes", "[]")
        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
        for i, outcome in enumerate(outcomes):
            key = _normalize_outcome(outcome)
            if key in result and i < len(clob_ids):
                result[key] = clob_ids[i]
    return result


async def analyze_market(
    market: dict[str, Any],
    positions: list[dict] | None = None,
    risk_tolerance: str | None = None,
) -> MarketAnalysis:
    """Analyze a market and return a structured recommendation."""
    from polymarket_agent.clob.client import (
        get_last_trade_price,
        get_midpoint,
        get_orderbook_summary,
        get_price_history,
    )
    from polymarket_agent.gamma.client import extract_market_signals

    # --- Extract token IDs ---
    # Gamma returns tokens in two formats:
    #   1. "tokens" list (from events endpoint, sometimes populated)
    #   2. "clobTokenIds" JSON string + "outcomes" list (from markets endpoint)
    yes_token_id = ""
    no_token_id = ""
    yes_price = 0.5
    no_price = 0.5

    tokens = market.get("tokens") or []
    if tokens:
        # Format 1: tokens list with outcome/token_id/price
        for tok in tokens:
            key = _normalize_outcome(tok.get("outcome", ""))
            price = float(tok.get("price", 0.5))
            if key == "YES":
                yes_price = price
                yes_token_id = tok.get("token_id", "")
            elif key == "NO":
                no_price = price
                no_token_id = tok.get("token_id", "")
    else:
        # Format 2: clobTokenIds + outcomes + outcomePrices (all JSON strings)
        clob_ids_raw = market.get("clobTokenIds", "[]")
        outcomes_raw = market.get("outcomes", "[]")
        prices_raw = market.get("outcomePrices", "[]")

        clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else (clob_ids_raw or [])
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])

        for i, outcome in enumerate(outcomes):
            tid = clob_ids[i] if i < len(clob_ids) else ""
            price = float(prices[i]) if i < len(prices) else 0.5
            key = _normalize_outcome(outcome)
            if key == "YES":
                yes_token_id = tid
                yes_price = price
            elif key == "NO":
                no_token_id = tid
                no_price = price

        logger.info(
            "Parsed from clobTokenIds: YES=%s... NO=%s... prices=%.4f/%.4f",
            yes_token_id[:20], no_token_id[:20], yes_price, no_price,
        )

    # --- Fetch LIVE prices from CLOB (Gamma prices are cached/stale) ---
    if yes_token_id:
        mid = get_midpoint(yes_token_id)
        if mid > 0:
            yes_price = mid
            no_price = round(1.0 - mid, 4)
            logger.info("Using CLOB midpoint: YES=%.4f NO=%.4f", yes_price, no_price)
        else:
            ltp = get_last_trade_price(yes_token_id)
            if ltp > 0:
                yes_price = ltp
                no_price = round(1.0 - ltp, 4)
                logger.info("Using CLOB last-trade: YES=%.4f NO=%.4f", yes_price, no_price)

    # Fetch enriched market data
    market_signals = extract_market_signals(market)

    price_history: list[dict[str, Any]] = []
    orderbook_summary: dict[str, Any] = {}
    if yes_token_id:
        # Use shorter interval for short-term markets (5min, 15min, 1h)
        end_date = market.get("endDate", "")
        history_interval = "1h"
        history_fidelity = 5
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                remaining_secs = (end_dt - datetime.now(timezone.utc)).total_seconds()
                if remaining_secs < 900:  # < 15 min
                    history_interval = "1m"
                    history_fidelity = 15
                elif remaining_secs < 3600:  # < 1 hour
                    history_interval = "5m"
                    history_fidelity = 12
                elif remaining_secs < 86400:  # < 1 day
                    history_interval = "30m"
                    history_fidelity = 12
            except (ValueError, TypeError):
                pass
        price_history = await get_price_history(
            yes_token_id, interval=history_interval, fidelity=history_fidelity
        )
        orderbook_summary = get_orderbook_summary(yes_token_id)

    logger.info(
        "Market signals for %s: signals=%s, history_points=%d, orderbook=%s",
        market.get("question", "?")[:60],
        market_signals,
        len(price_history),
        orderbook_summary,
    )

    # Retrieve past evaluations for this market
    market_key = market.get("conditionId") or market.get("condition_id") or market.get("question", "")
    past_evals = list(_evaluation_history.get(market_key, []))

    prompt = build_analysis_prompt(
        question=market.get("question", ""),
        description=market.get("description", ""),
        yes_price=yes_price,
        no_price=no_price,
        volume=float(market.get("volume", 0) or 0),
        liquidity=float(market.get("liquidity", 0) or 0),
        end_date=market.get("endDate", "unknown"),
        positions=positions,
        market_signals=market_signals,
        price_history=price_history,
        orderbook_summary=orderbook_summary,
        past_evaluations=past_evals,
    )

    system = build_system_prompt(risk_tolerance or settings.risk_tolerance)

    # Retry up to 3 times on empty or malformed JSON responses
    last_err: Exception | None = None
    for attempt in range(3):
        raw = await chat(system, prompt)
        if not raw or not raw.strip():
            logger.warning("LLM returned empty response (attempt %d/3)", attempt + 1)
            last_err = ValueError("LLM returned empty response")
            continue
        try:
            analysis = _parse_analysis(raw)
            break
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_err = exc
            logger.warning(
                "LLM returned invalid JSON (attempt %d/3): %s — raw: %s",
                attempt + 1, exc, raw[:200],
            )
    else:
        raise last_err  # type: ignore[misc]

    # Record this evaluation in history
    if market_key not in _evaluation_history:
        _evaluation_history[market_key] = deque(maxlen=MAX_HISTORY)
    _evaluation_history[market_key].append({
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "yes_price": yes_price,
        "no_price": no_price,
        "recommendation": analysis.recommendation,
        "confidence": analysis.confidence,
        "estimated_probability": analysis.estimated_probability,
        "reasoning": analysis.reasoning[:120],  # abbreviated
    })

    return analysis


async def parse_query(user_input: str) -> ParsedQuery:
    """Parse a natural language query into structured search parameters."""
    raw = await chat(QUERY_SYSTEM_PROMPT, user_input)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    data = json.loads(text)
    return ParsedQuery(**data)
