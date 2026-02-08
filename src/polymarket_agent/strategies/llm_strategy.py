"""Simplified strategy: LLM decides direction, confidence determines sizing.

Flow:
1. Local math computes: signal_strength, probability, volatility, trend
2. All data sent to LLM → LLM decides BUY_YES/BUY_NO/SKIP + confidence
3. Confidence determines bet size (tiered: 95%→max, 85%→70%, 75%→40%, 65%→20%)
4. Protections: no buying at $0.99+, cap by max_bet, cap by balance, min order size
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from polymarket_agent.analyst.llm import chat, extract_token_ids
from polymarket_agent.clob import client as clob
from polymarket_agent.config import settings
from polymarket_agent.strategies.base import TradeSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM decision prompt — emphasizes precision of confidence for sizing
# ---------------------------------------------------------------------------

_DECISION_SYSTEM_PROMPT = """\
You are an expert short-term crypto trader analyzing Polymarket up/down markets.

These markets resolve YES if the crypto spot price is ABOVE the reference price \
at expiry, and NO if BELOW. That's the ONLY thing that matters for resolution.

You are only called in the FINAL MINUTES before expiry, after the agent has \
accumulated extensive price data throughout the window. You have the full \
price history to analyze — use it to make the best possible decision.

You receive pre-computed mathematical analysis AND raw price trend data. \
The math is precise — your job is to add pattern recognition and judgment.

## YOUR CONFIDENCE CONTROLS THE BET SIZE
Your confidence directly determines how much money is wagered:
- confidence >= 0.95 → MAX BET (100% of max_bet)
- confidence >= 0.85 → LARGE BET (70% of max_bet)
- confidence >= 0.75 → MEDIUM BET (40% of max_bet)
- confidence >= 0.65 → SMALL BET (20% of max_bet)
- confidence < 0.65 → NO BET (automatic SKIP)

**Be VERY precise with your confidence.** 0.95 means "I'm almost certain this \
side wins." Only give 0.95+ if the lead is dominant and unlikely to reverse. \
0.75 means "probable but could go either way." 0.65 is "slight lean, small bet."

## What to analyze
1. **Price trend**: Is the price moving TOWARD or AWAY from the reference?
2. **Lead stability**: Is the lead growing, stable, or eroding?
3. **Reversal risk**: Given the volatility and time remaining, can the lead reverse?
4. **Return potential**: If YES costs $0.85, buying YES and winning = 18% return. \
If YES costs $0.95, buying YES and winning = only 5% return. Consider this.
5. **Coinflip detection**: If the price is very close to reference with high volatility, \
this is a coinflip — SKIP.

## When to give HIGH confidence (0.85+)
- Price clearly on one side of reference with a stable or growing lead
- Signal strength >= 2x with time running out (hard to reverse)
- Strong, consistent trend reinforcing the current direction

## When to give LOW confidence (0.65-0.75)
- Price is on one side but the lead is small or volatile
- Signal strength 0.5-1x — could go either way
- Mixed signals: price on one side but trending toward reference

## When to SKIP
- Price oscillating around reference with no clear direction
- Signal strength < 0.5x — essentially a coinflip
- Choppy, directionless price action
- You cannot determine which side will win with >65% confidence

## Important rules
- A pullback WHILE STILL ON THE WINNING SIDE is NOT a reversal. \
Example: spot goes from ref+$200 to ref+$150 — that's still UP, not a sell signal.
- $50 move in BTC ($97,000) is 0.05% — that's noise, not a signal.
- Near expiry with a clear lead → BE MORE CONFIDENT, not less. Less time = less reversal risk.
- Missing a good trade is costly. Only SKIP if genuinely uncertain (<65%).
- The pre-computed probability is your baseline. Adjust based on trend, but don't ignore the math.

Respond with ONLY valid JSON (no markdown, no extra text):
{
  "recommendation": "BUY_YES" | "BUY_NO" | "SKIP",
  "confidence": 0.0 to 1.0,
  "reasoning": "1-2 sentences explaining your decision"
}\
"""


# ---------------------------------------------------------------------------
# Core math functions (informational — sent to LLM, not used as gates)
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_probability(signal_strength: float, direction: str) -> float:
    """Compute P(YES wins) from signal strength and price direction."""
    if direction == "UP":
        return _normal_cdf(signal_strength)
    else:
        return _normal_cdf(-signal_strength)


def compute_signal_strength(
    price_tracker_data: list[dict],
    price_diff: float,
    time_remaining_secs: float,
) -> float | None:
    """Compute signal strength from observed price volatility.

    signal_strength = |price_diff| / (sigma * sqrt(steps_remaining))
    Returns None if insufficient data.
    """
    if not price_tracker_data or len(price_tracker_data) < 3:
        return None
    if time_remaining_secs <= 0:
        return float("inf")

    spots = [s["spot_price"] for s in price_tracker_data]
    n = len(spots)

    consec_changes = [abs(spots[i] - spots[i - 1]) for i in range(1, n)]
    avg_change = sum(consec_changes) / len(consec_changes)

    if avg_change <= 0:
        return 10.0  # Price hasn't moved — very stable

    sample_interval_s = 10.0
    if n >= 2:
        try:
            t0 = datetime.strptime(price_tracker_data[0]["timestamp"], "%H:%M:%S")
            t1 = datetime.strptime(price_tracker_data[-1]["timestamp"], "%H:%M:%S")
            span_s = (t1 - t0).total_seconds()
            if span_s > 0:
                sample_interval_s = span_s / (n - 1)
        except (KeyError, ValueError):
            pass

    steps_per_min = 60.0 / sample_interval_s if sample_interval_s > 0 else 6
    steps_remaining = (time_remaining_secs / 60) * steps_per_min

    if steps_remaining <= 0:
        return 10.0

    expected_swing = avg_change * math.sqrt(steps_remaining)
    return abs(price_diff) / expected_swing if expected_swing > 0 else 10.0


def _compute_volatility_stats(price_tracker_data: list[dict]) -> dict | None:
    """Compute volatility stats from price tracker data for the LLM prompt."""
    if not price_tracker_data or len(price_tracker_data) < 3:
        return None

    spots = [s["spot_price"] for s in price_tracker_data]
    n = len(spots)
    hi, lo = max(spots), min(spots)

    consec_changes = [abs(spots[i] - spots[i - 1]) for i in range(1, n)]
    avg_change = sum(consec_changes) / len(consec_changes)
    max_change = max(consec_changes)

    return {
        "hi": hi,
        "lo": lo,
        "max_swing": hi - lo,
        "avg_change": avg_change,
        "max_consec_change": max_change,
        "samples": n,
    }


# ---------------------------------------------------------------------------
# Confidence → bet size mapping
# ---------------------------------------------------------------------------

def _confidence_to_size_fraction(confidence: float) -> float | None:
    """Map LLM confidence to fraction of max_bet_usdc.

    Returns None if confidence is too low to trade.
    """
    if confidence >= 0.95:
        return 1.00  # 100% of max_bet
    elif confidence >= 0.85:
        return 0.70  # 70% of max_bet
    elif confidence >= 0.75:
        return 0.40  # 40% of max_bet
    elif confidence >= 0.65:
        return 0.20  # 20% of max_bet
    else:
        return None  # Too low — skip


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _build_decision_prompt(
    question: str,
    time_remaining_secs: float,
    spot_price_info: dict,
    computed: dict,
    price_tracker_data: list[dict] | None,
    volatility: dict | None,
    yes_price: float,
    no_price: float,
) -> str:
    """Build a focused prompt with all pre-computed data for the LLM."""
    mins = int(time_remaining_secs // 60)
    secs = int(time_remaining_secs % 60)

    diff = spot_price_info["price_diff"]
    position = "ABOVE" if diff > 0 else "BELOW"
    favors = "YES/Up" if diff > 0 else "NO/Down"

    # Compute potential returns
    yes_return = ((1.0 - yes_price) / yes_price * 100) if yes_price > 0 else 0
    no_return = ((1.0 - no_price) / no_price * 100) if no_price > 0 else 0

    parts = [
        f"## Market: {question}",
        f"**Time remaining:** {mins}m {secs}s",
        "",
        f"## Current Position",
        f"- Spot price: ${spot_price_info['spot_price']:,.2f}",
        f"- Reference price: ${spot_price_info['reference_price']:,.2f}",
        f"- Difference: ${diff:+,.2f} ({spot_price_info.get('price_diff_pct', 0) or 0:+.3f}%)",
        f"- Position: {position} reference → **{favors} is winning**",
        "",
        f"## Market Prices & Potential Returns",
        f"- YES price: ${yes_price:.4f} → if YES wins: **{yes_return:.1f}% return**",
        f"- NO price: ${no_price:.4f} → if NO wins: **{no_return:.1f}% return**",
        "",
        f"## Pre-computed Math (informational)",
        f"- Signal strength: {computed['signal_strength']:.2f}x expected random-walk swing",
        f"- Computed P(YES wins): {computed['prob_yes']:.1%}",
        f"- Computed P(NO wins): {1 - computed['prob_yes']:.1%}",
    ]

    if volatility:
        parts.extend([
            "",
            f"## Observed Volatility ({volatility['samples']} samples)",
            f"- Avg change between samples: ${volatility['avg_change']:,.2f}",
            f"- Max single-sample jump: ${volatility['max_consec_change']:,.2f}",
            f"- Price range: ${volatility['lo']:,.2f} – ${volatility['hi']:,.2f} "
            f"(swing: ${volatility['max_swing']:,.2f})",
        ])

    if price_tracker_data:
        recent = price_tracker_data[-20:]  # Last 20 samples
        parts.extend(["", f"## Price Trend ({len(recent)} recent samples, newest last)"])
        for s in recent:
            diff_str = ""
            if s.get("diff") is not None:
                diff_str = f" (ref{s['diff']:+,.2f})"
            parts.append(f"  {s['timestamp']} ${s['spot_price']:,.2f}{diff_str}")

        # Trend summary
        spots = [s["spot_price"] for s in recent]
        first_s, last_s = spots[0], spots[-1]
        delta = last_s - first_s
        trend_dir = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"

        # Consecutive streak
        streak_count = 0
        streak_dir = ""
        for i in range(len(spots) - 1, 0, -1):
            d = "up" if spots[i] > spots[i - 1] else "down" if spots[i] < spots[i - 1] else ""
            if not d:
                continue
            if streak_count == 0:
                streak_dir = d
                streak_count = 1
            elif d == streak_dir:
                streak_count += 1
            else:
                break

        streak_str = f", {streak_count} consecutive {streak_dir}" if streak_count else ""
        parts.append(f"  **Trend: {trend_dir} ${delta:+,.2f}{streak_str}**")

    parts.extend([
        "",
        "## Sizing Reminder",
        "Your confidence directly controls bet size:",
        "  0.95+ = MAX BET | 0.85+ = LARGE | 0.75+ = MEDIUM | 0.65+ = SMALL | <0.65 = SKIP",
        "Be precise. Only give 0.95+ if you're almost certain.",
        "",
        "Analyze the trend, volatility, position, and returns. Make your trading decision.",
    ])

    return "\n".join(parts)


def _parse_decision(raw: str) -> dict:
    """Parse LLM JSON response into a decision dict."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt repair
        repaired = text.rstrip()
        if repaired.endswith("\\"):
            repaired = repaired[:-1]
        quote_count = repaired.count('"') - repaired.count('\\"')
        if quote_count % 2 == 1:
            repaired += '"'
        repaired = repaired.rstrip(",")
        open_braces = repaired.count("{") - repaired.count("}")
        repaired += "}" * open_braces
        data = json.loads(repaired)

    conf = data.get("confidence", 0.5)
    if isinstance(conf, (int, float)) and conf > 1.0:
        conf /= 100.0

    return {
        "recommendation": data.get("recommendation", "SKIP"),
        "confidence": conf,
        "reasoning": data.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class LLMStrategy:
    """Simplified strategy: LLM decides direction, confidence determines sizing."""

    async def evaluate(
        self,
        market: dict[str, Any],
        positions: list[dict[str, Any]],
        price_tracker_data: list[dict] | None = None,
        reasoning_chain: list[dict] | None = None,
        time_remaining_secs: float | None = None,
        bankroll: float | None = None,
        spot_price_info: dict | None = None,
    ) -> tuple[TradeSignal | None, dict]:
        """Evaluate a market: compute math locally, ask LLM to decide, size by confidence.

        Returns (signal, info_dict) where info_dict has debug/display info.
        """
        info: dict[str, Any] = {
            "recommendation": "SKIP",
            "confidence": 0.0,
            "reasoning": "",
            "probability_yes": 0.5,
            "signal_strength": None,
        }

        # --- Need spot price with reference ---
        if not spot_price_info or spot_price_info.get("price_diff") is None:
            info["reasoning"] = "Waiting for spot + reference price data"
            return None, info

        # --- Need time context ---
        if time_remaining_secs is None or time_remaining_secs <= 0:
            info["reasoning"] = "Market expired or no time data"
            return None, info

        # --- Trade window gate: only trade in the last N seconds ---
        # Before this, we accumulate price data but don't call the LLM or trade.
        if time_remaining_secs > settings.trade_window_secs:
            mins_to_window = (time_remaining_secs - settings.trade_window_secs) / 60
            info["reasoning"] = (
                f"Accumulating data — trade window opens in {mins_to_window:.1f}m "
                f"(last {settings.trade_window_secs}s)"
            )
            return None, info

        price_diff = spot_price_info["price_diff"]
        direction = "UP" if price_diff > 0 else "DOWN"

        # =====================================================================
        # PHASE 1: Local math computation (informational — for LLM context)
        # =====================================================================

        sig_strength = compute_signal_strength(
            price_tracker_data or [], price_diff, time_remaining_secs,
        )

        # Fallback for insufficient samples
        if sig_strength is None:
            diff_pct = abs(spot_price_info.get("price_diff_pct", 0) or 0)
            if diff_pct < 0.005:
                info["reasoning"] = (
                    f"Need more data ({len(price_tracker_data or [])} samples, "
                    f"diff={diff_pct:.4f}%)"
                )
                return None, info
            sig_strength = 0.7

        info["signal_strength"] = sig_strength

        # Probability from normal CDF (informational)
        prob_yes = compute_probability(sig_strength, direction)
        info["probability_yes"] = prob_yes

        math_recommendation = "BUY_YES" if direction == "UP" else "BUY_NO"

        # Get market prices for both sides
        token_ids = extract_token_ids(market)
        yes_token_id = token_ids.get("YES", "")
        no_token_id = token_ids.get("NO", "")

        # Get live prices via midpoint (fast)
        yes_mid = clob.get_midpoint(yes_token_id) if yes_token_id else 0
        no_mid = clob.get_midpoint(no_token_id) if no_token_id else 0
        if yes_mid > 0 and no_mid <= 0:
            no_mid = round(1.0 - yes_mid, 4)
        elif no_mid > 0 and yes_mid <= 0:
            yes_mid = round(1.0 - no_mid, 4)

        # Volatility stats for prompt
        volatility = _compute_volatility_stats(price_tracker_data or [])

        computed = {
            "signal_strength": sig_strength,
            "prob_yes": prob_yes,
        }

        # =====================================================================
        # PHASE 2: LLM decision (the LLM is the boss)
        # =====================================================================

        prompt = _build_decision_prompt(
            question=market.get("question", "?"),
            time_remaining_secs=time_remaining_secs,
            spot_price_info=spot_price_info,
            computed=computed,
            price_tracker_data=price_tracker_data,
            volatility=volatility,
            yes_price=yes_mid,
            no_price=no_mid,
        )

        # Call LLM with retry
        decision = None
        for attempt in range(3):
            try:
                raw = await chat(_DECISION_SYSTEM_PROMPT, prompt)
                if raw and raw.strip():
                    decision = _parse_decision(raw)
                    break
            except Exception as exc:
                logger.warning("LLM call failed (attempt %d/3): %s", attempt + 1, exc)

        # Fallback if LLM is unavailable: use math-only decision
        if decision is None:
            logger.warning("LLM unavailable — falling back to math-only decision")
            math_win_prob = prob_yes if direction == "UP" else 1.0 - prob_yes
            decision = {
                "recommendation": math_recommendation,
                "confidence": math_win_prob,
                "reasoning": "LLM unavailable, using math-only",
            }

        info["recommendation"] = decision["recommendation"]
        info["confidence"] = decision["confidence"]
        info["reasoning"] = decision["reasoning"]
        info["llm_decision"] = decision

        # =====================================================================
        # PHASE 3: Size by confidence and execute
        # =====================================================================

        if decision["recommendation"] == "SKIP":
            return None, info

        # Confidence gate (< 0.65 = skip)
        size_fraction = _confidence_to_size_fraction(decision["confidence"])
        if size_fraction is None:
            info["reasoning"] = (
                f"LLM says {decision['recommendation']} but confidence too low: "
                f"{decision['confidence']:.0%} < 65% — {decision['reasoning']}"
            )
            return None, info

        # Determine which side to trade
        if decision["recommendation"] == "BUY_YES":
            token_id = yes_token_id
        elif decision["recommendation"] == "BUY_NO":
            token_id = no_token_id
        else:
            info["reasoning"] = f"Unknown recommendation: {decision['recommendation']}"
            return None, info

        if not token_id:
            info["reasoning"] = f"No token ID for {decision['recommendation']}"
            return None, info

        # Get ACTUAL price from the YES token's orderbook (primary liquidity).
        # In Polymarket binary markets, the NO token's orderbook is often empty
        # or has inflated asks. Real NO trades execute via complement matching
        # against the YES orderbook:
        #   BUY_YES  → actual_price = YES best_ask
        #   BUY_NO   → actual_price = 1 - YES best_bid  (complement)
        if not yes_token_id:
            info["reasoning"] = "No YES token ID for orderbook lookup"
            return None, info

        yes_ob = clob.get_orderbook_summary(yes_token_id)
        if not yes_ob:
            info["reasoning"] = "No orderbook data available"
            return None, info

        min_order_size = yes_ob.get("min_order_size", 5.0)

        if decision["recommendation"] == "BUY_YES":
            if yes_ob.get("best_ask") is None or yes_ob["best_ask"] <= 0:
                info["reasoning"] = "No asks in YES orderbook"
                return None, info
            actual_price = yes_ob["best_ask"]
        else:  # BUY_NO
            if yes_ob.get("best_bid") is None or yes_ob["best_bid"] <= 0:
                info["reasoning"] = "No bids in YES orderbook (can't derive NO price)"
                return None, info
            actual_price = round(1.0 - yes_ob["best_bid"], 4)

        logger.info(
            "Price for %s: actual=%.4f (YES ob: bid=%.4f ask=%.4f)",
            decision["recommendation"], actual_price,
            yes_ob.get("best_bid", 0), yes_ob.get("best_ask", 0),
        )

        # Protection: don't buy at $0.99+ (return < 1%, not worth the risk)
        if actual_price >= 0.99:
            info["reasoning"] = (
                f"Price too high: ${actual_price:.4f} — return < 1%, not worth the risk"
            )
            return None, info

        # --- Confidence-based sizing ---
        size_usd = settings.max_bet_usdc * size_fraction

        # Convert to shares
        size_shares = size_usd / actual_price if actual_price > 0 else 0

        # Cap by available balance
        if bankroll is None:
            bankroll = clob.get_balance()
        if bankroll is not None and bankroll < size_shares * actual_price:
            size_shares = bankroll / actual_price if actual_price > 0 else 0

        # Enforce minimum order size — bump up if affordable
        if size_shares < min_order_size:
            min_cost = min_order_size * actual_price
            if min_cost <= settings.max_bet_usdc and (bankroll is None or min_cost <= bankroll):
                size_shares = min_order_size
            else:
                info["reasoning"] = (
                    f"Size {size_shares:.1f} < min {min_order_size:.0f} "
                    f"(cost ${min_cost:.2f} > budget)"
                )
                return None, info

        # Check portfolio exposure
        current_exposure = sum(
            float(p.get("size", 0) or 0) * float(p.get("currentPrice", 0) or 0)
            for p in positions
        )
        if current_exposure + (size_shares * actual_price) > settings.max_portfolio_usdc:
            remaining = settings.max_portfolio_usdc - current_exposure
            if remaining > 0 and remaining / actual_price >= min_order_size:
                size_shares = remaining / actual_price
            else:
                info["reasoning"] = "Portfolio limit reached"
                return None, info

        # Cap by max_bet in shares
        max_shares = settings.max_bet_usdc / actual_price if actual_price > 0 else 0
        if size_shares > max_shares:
            size_shares = max_shares

        potential_return = (1.0 - actual_price) / actual_price * 100 if actual_price > 0 else 0
        trade_reasoning = (
            f"LLM: {decision['recommendation']} ({decision['confidence']:.0%}) | "
            f"Size: ${size_shares * actual_price:.2f} ({size_fraction:.0%} of max) | "
            f"Ask=${actual_price:.4f} Return={potential_return:.1f}% | "
            f"Math: {direction} {sig_strength:.1f}x P={prob_yes:.1%} | "
            f"{decision['reasoning']}"
        )
        info["reasoning"] = trade_reasoning

        return TradeSignal(
            action=decision["recommendation"],
            token_id=token_id,
            price=actual_price,
            size=round(size_shares, 1),
            confidence=decision["confidence"],
            reasoning=trade_reasoning,
        ), info
