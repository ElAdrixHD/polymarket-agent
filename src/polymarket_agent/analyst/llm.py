"""Multi-provider LLM client for market analysis."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from polymarket_agent.analyst.prompts import (
    QUERY_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_system_prompt,
)
from polymarket_agent.config import settings


class ParsedQuery(BaseModel):
    search_terms: list[str]
    max_hours: float | None = None
    action: str = "search"  # "search" or "analyze"
    risk_tolerance: str | None = None


class MarketAnalysis(BaseModel):
    recommendation: str  # BUY_YES, BUY_NO, SKIP
    confidence: float
    estimated_probability: float = 0.5
    reasoning: str
    suggested_price: float | None = None
    suggested_size: float | None = None


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
    )
    return resp.choices[0].message.content or ""


async def _call_anthropic(system: str, user: str) -> str:
    """Call LLM via Anthropic SDK."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
    )
    resp = await client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


async def chat(system: str, user: str) -> str:
    """Send a chat message to the configured LLM provider."""
    if settings.llm_provider == "anthropic":
        return await _call_anthropic(system, user)
    return await _call_openai(system, user)


def _parse_analysis(raw: str) -> MarketAnalysis:
    """Parse LLM JSON response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        # Remove ```json ... ``` wrapper
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    data = json.loads(text)
    return MarketAnalysis(**data)


async def analyze_market(
    market: dict[str, Any],
    positions: list[dict] | None = None,
    risk_tolerance: str | None = None,
) -> MarketAnalysis:
    """Analyze a market and return a structured recommendation."""
    tokens = market.get("tokens", [])
    yes_price = 0.5
    no_price = 0.5
    for tok in tokens:
        outcome = tok.get("outcome", "").upper()
        price = float(tok.get("price", 0.5))
        if outcome == "YES":
            yes_price = price
        elif outcome == "NO":
            no_price = price

    prompt = build_analysis_prompt(
        question=market.get("question", ""),
        description=market.get("description", ""),
        yes_price=yes_price,
        no_price=no_price,
        volume=float(market.get("volume", 0) or 0),
        liquidity=float(market.get("liquidity", 0) or 0),
        end_date=market.get("endDate", "unknown"),
        positions=positions,
    )

    system = build_system_prompt(risk_tolerance or settings.risk_tolerance)
    raw = await chat(system, prompt)
    return _parse_analysis(raw)


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
