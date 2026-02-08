"""Prompt templates for LLM market analysis."""

import math
from datetime import datetime, timezone

QUERY_SYSTEM_PROMPT = """\
You are a query interpreter for a Polymarket prediction market agent.
The user will ask you to find or analyze specific markets in natural language.

Extract the search parameters from the user's request and respond with valid JSON:
{
  "search_terms": ["keyword1", "keyword2"],
  "max_hours": number or null,
  "action": "search" | "analyze"
}

Rules:
- search_terms: keywords to search for on Polymarket. Include synonyms and related terms.
  For crypto: include both full name and ticker (e.g. ["bitcoin", "btc"])
  For short-term markets: include the timeframe keyword (e.g. ["btc", "bitcoin", "up or down", "15min"])
- max_hours: if the user mentions a timeframe like "15min", "1h", "24h", "this week", convert to hours (0.25, 1, 24, 168). null if no time constraint.
  IMPORTANT: If the user says "btc 15min" or "bitcoin 1h" they likely mean the "Up or Down" short-term \
  prediction markets (e.g. "Bitcoin Up or Down - 9:15AM-9:30AM"), NOT a time filter. Set max_hours to null \
  and include the timeframe as a search term instead (e.g. "15min", "5min", "1h").
  Only set max_hours when the user explicitly says "expiring in", "closing in", "ending within", etc.
- action: "search" if user just wants to see/list markets, "analyze" if they want analysis/recommendations
Polymarket has these short-term "Up or Down" markets for crypto (BTC, ETH, SOL, XRP) in 5min, 15min, and 1h intervals. \
Their slugs look like: btc-updown-15m-{timestamp}, eth-updown-5m-{timestamp}, etc.

Examples:
- "show me btc bets expiring in 1h" → {"search_terms": ["bitcoin", "btc"], "max_hours": 1, "action": "search"}
- "analyze btc 15min" → {"search_terms": ["bitcoin", "btc", "up or down", "15"], "max_hours": null, "action": "analyze"}
- "analyze crypto markets closing in 15min" → {"search_terms": ["crypto", "bitcoin", "ethereum"], "max_hours": 0.25, "action": "analyze"}
- "what political bets are there?" → {"search_terms": ["politics", "trump", "election"], "max_hours": null, "action": "search"}
- "btc 1h markets" → {"search_terms": ["bitcoin", "btc", "up or down", "1h"], "max_hours": null, "action": "search"}
- "eth 5min" → {"search_terms": ["ethereum", "eth", "up or down", "5"], "max_hours": null, "action": "search"}

Respond with JSON only.\
"""

SYSTEM_PROMPT = """\
You are an expert prediction market analyst and short-term trader. Your job is to \
evaluate Polymarket prediction markets and recommend whether to buy YES tokens, \
buy NO tokens, or skip.

**CURRENT TIME: {current_time} UTC**

You must be calibrated: if you think the true probability is 60%, and YES trades at \
50 cents, that's a BUY_YES. If YES trades at 70 cents, that's a BUY_NO (or SKIP).

## Trading Signals You Will Receive
You will be given real market data to inform your decision:
- **Price history**: recent price points showing the trend/momentum
- **Order book summary**: bid/ask depth and ratio indicating buying/selling pressure
- **Market signals**: 1h and 24h price changes, spread, volume

## How to Use the Data
- **Spot price (CRITICAL for Up/Down markets)**: For crypto up/down markets you will receive the \
LIVE spot price from Binance and the market's reference price. This is your MOST IMPORTANT signal:
  - If spot > reference → asset is UP → YES should win → BUY_YES
  - If spot < reference → asset is DOWN → NO should win → BUY_NO
  - The magnitude of the difference relative to **observed volatility** and time remaining determines confidence.
  - **Check the "Dynamic Risk Assessment" section** — it shows how much the price has actually \
swung during the observation window. A $300 lead means nothing if the price swung $500 in the last 5 minutes.
  - **This is real external data, not market opinion. Trust it over order book sentiment.**
  - **Chainlink vs Binance**: These markets resolve using the Chainlink oracle price feed. Chainlink \
aggregates prices from multiple exchanges including Binance. In practice, Chainlink BTC/USD and \
Binance BTCUSDT differ by only a few dollars (< 0.01%). **Treat the Binance spot price as a reliable \
proxy for the Chainlink resolution price.** Do NOT skip trades due to the Chainlink/Binance distinction.
- **Momentum**: If price has been trending up (series of higher prices), that suggests \
YES is gaining. If trending down, NO side is gaining.
- **Order book pressure**: bid_ask_ratio > 1 means more buying pressure (bullish for YES). \
ratio < 1 means more selling pressure (bearish for YES).
- **Spread**: A tight spread means the market is liquid and prices are reliable. \
A wide spread means less certainty.
- **Volume**: Higher volume means more information is priced in. Low volume markets may \
have more mispricing opportunities.
- **Price changes**: 1h and 24h changes show recent momentum direction and magnitude.
- **Past evaluations**: You may receive your own previous analyses for this market \
(from prior evaluation cycles). Use them to track how the market has evolved:
  - If the price is moving toward your estimated probability, your thesis is confirmed.
  - If the price moved away, reconsider your thesis.
  - If you previously said SKIP but prices have shifted significantly, re-evaluate.
  - Look for acceleration or deceleration in price changes across evaluations.

## Existing Orders (Session Memory)
If you have already placed orders on this market during this session:
- **Do NOT duplicate** the same position unless the edge has grown significantly since \
your last order (price moved further in your favor).
- **Consider the total exposure**: your new order + existing orders should not over-expose \
on a single market.
- If the market moved against your previous order, be cautious — do not average down \
unless you have very high confidence.
- If the market moved in your favor, consider whether the edge is now too small to add more.

## Short-Term Markets (Up or Down) — CRITICAL RULES
These are binary markets that resolve based on whether the spot price is ABOVE or BELOW \
a reference price at a specific time. **The ONLY thing that matters is the final price \
relative to the reference.**

### What determines the outcome
- The market resolves YES if the spot price is ABOVE the reference price at expiry.
- The market resolves NO if the spot price is BELOW the reference price at expiry.
- **A small dip while still above reference does NOT mean NO will win.** Example: if \
reference is $69,000 and spot goes from $69,200 → $69,100, that is still UP. Do NOT \
interpret this as "reversal" or "momentum shift toward NO".

### How to decide
- **POSITION vs REFERENCE is your #1 signal**: Is the spot price currently above or below \
the reference? By how much? This is more important than any short-term momentum.
- **Use observed volatility to judge reversal probability**: Check the Dynamic Risk Assessment. \
If the current lead (|spot - reference|) is larger than what the observed volatility suggests \
can reverse in the remaining time, that's a strong signal.
- **Micro-movements are NOT momentum reversals**: The spot price fluctuates constantly. A \
$50-100 pullback in BTC while still $150 above reference is normal noise, NOT a bearish signal. \
Only consider it a reversal risk if the price is approaching or crossing the reference.
- **Time decay matters**: as resolution approaches, if the price is firmly on one side of the \
reference, the probability of it staying there increases (less time for reversal).
- **SKIP = guaranteed loss of opportunity**: these markets expire quickly, so being in the market \
with even a small edge is better than missing it entirely.
- **Lower your edge threshold**: for markets expiring in < 30 minutes, an edge of 2 cents is actionable.
- **Near-certain outcomes are STILL worth trading**: If signal strength is >= 2x and the outcome is \
nearly certain (e.g. 97%+ probability), BUY even if the market price is close to the true probability. \
A 2-3 cent edge on a near-certain outcome is FREE MONEY. Do NOT skip these. Example: if NO is ~97% \
likely and NO trades at $0.95, that's a 2-cent edge with 97% certainty — BUY_NO, do NOT skip.

### Common mistakes to AVOID
- **DO NOT BUY_NO just because the price dipped slightly while still above reference.** \
If spot is $69,100 and reference is $69,000, BUY_YES is correct even if spot just dropped from $69,200.
- **DO NOT confuse market token price momentum with spot price position.** The YES token \
price may fluctuate, but what matters is where the SPOT PRICE is relative to the REFERENCE.
- **DO NOT overthink momentum**: a $50 pullback in a $69,000 asset is 0.07% — that's noise.

## Decision Framework
1. **Position check**: Is spot above or below reference? By how much?
2. **Volatility check**: Can the observed volatility reverse this lead in the remaining time?
3. **Order book**: Does bid/ask pressure confirm or contradict?
4. Estimate the true probability based on all available signals
5. Compare to current price to find edges >= 3 cents (2 cents for short-term markets)
6. Only SKIP if you truly cannot form an opinion or the edge is < 2 cents

Always respond with ONLY valid JSON (no markdown, no extra text) matching this schema:
{
  "recommendation": "BUY_YES" | "BUY_NO" | "SKIP",
  "confidence": 0.0 to 1.0,
  "estimated_probability": 0.0 to 1.0 (probability that YES wins, NOT the recommended side),
  "reasoning": "short explanation, max 2 sentences",
  "suggested_price": 0.0 to 1.0 or null,
  "suggested_size": number or null
}

CRITICAL: "estimated_probability" is ALWAYS the probability that YES/Up wins.
- If you recommend BUY_YES because YES is likely, estimated_probability should be HIGH (e.g. 0.85)
- If you recommend BUY_NO because NO is likely, estimated_probability should be LOW (e.g. 0.05)
- Example: BTC is $300 below reference → NO will win → estimated_probability = 0.02 (YES has only 2% chance)

BE PRECISE with extreme probabilities — the difference between 0.03 and 0.005 matters for edge calculations:
- Signal strength >= 2x with < 5 min left → the side with the lead has > 99% probability. Use 0.995+ or 0.005-.
- Signal strength >= 3x with < 3 min left → use 0.998+ or 0.002-.
- Signal strength >= 5x → use 0.999+ or 0.001-.
- Do NOT round to 0.95 or 0.97 when the math says 99.5%+. Precision at the extremes is critical.
"""

def build_system_prompt() -> str:
    """Build the full system prompt with hardcoded risk profile."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    base = SYSTEM_PROMPT.replace("{current_time}", now)
    risk_block = (
        "\n## Risk Profile (Hardcoded)\n"
        "You are optimized for 15-minute crypto up/down markets.\n\n"
        "- Prioritize capital preservation over maximizing edge.\n"
        "- Only recommend trades with high confidence and clear signals.\n"
        "- A 5% safe gain is better than a 20% risky one.\n"
        "- Prefer entering late when the outcome direction is clear over early speculative entries.\n"
    )
    return base + risk_block


def compute_signal_strength(
    spot_price_info: dict | None,
    price_tracker_data: list[dict] | None,
    end_date: str,
) -> float | None:
    """Compute signal strength from observed volatility data.

    Returns the signal strength multiplier (e.g. 2.5x expected swing),
    or None if there isn't enough data to compute it.
    """
    if not price_tracker_data or len(price_tracker_data) < 3:
        return None
    if not spot_price_info or spot_price_info.get("price_diff") is None:
        return None

    now = datetime.now(timezone.utc)
    total_seconds = 0.0
    if end_date and end_date != "unknown":
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            total_seconds = (end_dt - now).total_seconds()
        except (ValueError, TypeError):
            pass
    if total_seconds <= 0:
        return None

    spots = [s["spot_price"] for s in price_tracker_data]
    n = len(spots)
    consec_changes = [abs(spots[i] - spots[i - 1]) for i in range(1, n)]
    avg_change = sum(consec_changes) / len(consec_changes)

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

    sigma_per_step = avg_change
    steps_per_min = 60.0 / sample_interval_s if sample_interval_s > 0 else 6
    mins_left = total_seconds / 60
    steps_remaining = mins_left * steps_per_min
    expected_swing = sigma_per_step * math.sqrt(steps_remaining) if steps_remaining > 0 else 0

    abs_diff = abs(spot_price_info["price_diff"])
    if expected_swing <= 0:
        return None

    return abs_diff / expected_swing


def build_analysis_prompt(
    question: str,
    description: str,
    yes_price: float,
    no_price: float,
    volume: float,
    liquidity: float,
    end_date: str,
    positions: list[dict] | None = None,
    market_signals: dict | None = None,
    price_history: list[dict] | None = None,
    orderbook_summary: dict | None = None,
    past_evaluations: list[dict] | None = None,
    order_history: list[dict] | None = None,
    spot_price_info: dict | None = None,
    price_tracker_data: list[dict] | None = None,
    reasoning_chain: list[dict] | None = None,
) -> str:
    """Build the user prompt for market analysis."""
    # Calculate time remaining
    now = datetime.now(timezone.utc)
    time_remaining_str = ""
    total_seconds = 0.0
    if end_date and end_date != "unknown":
        try:
            # Parse ISO format end date
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            remaining = end_dt - now
            total_seconds = remaining.total_seconds()
            if total_seconds <= 0:
                time_remaining_str = "**EXPIRED**"
            elif total_seconds < 60:
                time_remaining_str = f"**Time remaining:** {int(total_seconds)}s"
            elif total_seconds < 3600:
                time_remaining_str = f"**Time remaining:** {int(total_seconds // 60)}m {int(total_seconds % 60)}s"
            elif total_seconds < 86400:
                hours = int(total_seconds // 3600)
                mins = int((total_seconds % 3600) // 60)
                time_remaining_str = f"**Time remaining:** {hours}h {mins}m"
            else:
                days = int(total_seconds // 86400)
                time_remaining_str = f"**Time remaining:** {days} days"
        except (ValueError, TypeError):
            pass

    parts = [
        f"## Market Analysis Request",
        f"**Question:** {question}",
        f"**Description:** {description}" if description else "",
        time_remaining_str,
        f"**Current YES price:** ${yes_price:.2f}",
        f"**Current NO price:** ${no_price:.2f}",
        f"**Volume:** ${volume:,.0f}",
        f"**Liquidity:** ${liquidity:,.0f}",
        f"**End date:** {end_date}",
    ]

    # Spot price context for up/down crypto markets
    if spot_price_info:
        parts.append(f"\n## Live {spot_price_info['asset']} Spot Price (from Binance)")
        parts.append(f"- **Current spot price:** ${spot_price_info['spot_price']:,.2f}")
        if spot_price_info.get("reference_price"):
            ref = spot_price_info["reference_price"]
            diff = spot_price_info.get("price_diff", 0)
            pct = spot_price_info.get("price_diff_pct", 0)
            parts.append(f"- **Market reference price:** ${ref:,.2f}")
            parts.append(f"- **Difference:** ${diff:+,.2f} ({pct:+.3f}%)")
            if diff > 0:
                parts.append(f"- **Direction:** {spot_price_info['asset']} is ABOVE the reference → favors YES/Up")
            elif diff < 0:
                parts.append(f"- **Direction:** {spot_price_info['asset']} is BELOW the reference → favors NO/Down")
            else:
                parts.append(f"- **Direction:** {spot_price_info['asset']} is AT the reference → coin flip")

    # Dynamic risk guidance based on OBSERVED volatility from price tracker.
    # No static fallback — if there aren't enough price samples yet, the
    # risk assessment section is simply omitted until data accumulates.

    # Compute observed volatility from accumulated price snapshots
    observed_vol: dict[str, float] | None = None
    if price_tracker_data and len(price_tracker_data) >= 3:
        spots = [s["spot_price"] for s in price_tracker_data]
        n = len(spots)
        hi, lo = max(spots), min(spots)
        max_swing = hi - lo

        # Consecutive changes (absolute)
        consec_changes = [abs(spots[i] - spots[i - 1]) for i in range(1, n)]
        avg_change = sum(consec_changes) / len(consec_changes)
        max_consec_change = max(consec_changes)

        # Standard deviation of prices
        mean_price = sum(spots) / n
        variance = sum((p - mean_price) ** 2 for p in spots) / n
        std_dev = math.sqrt(variance)

        # Estimate time span between samples (use timestamps if available)
        sample_interval_s = 10.0  # default
        if len(price_tracker_data) >= 2:
            try:
                t0 = datetime.strptime(price_tracker_data[0]["timestamp"], "%H:%M:%S")
                t1 = datetime.strptime(price_tracker_data[-1]["timestamp"], "%H:%M:%S")
                span_s = (t1 - t0).total_seconds()
                if span_s > 0:
                    sample_interval_s = span_s / (n - 1)
            except (KeyError, ValueError):
                pass

        total_span_s = sample_interval_s * (n - 1)

        # Volatility per sample step (sigma per step, random walk model)
        # avg_change ≈ sigma per step.  To project over N future steps:
        #   expected_swing = sigma * sqrt(N)   (NOT sigma * N)
        sigma_per_step = avg_change
        steps_per_min = 60.0 / sample_interval_s if sample_interval_s > 0 else 6

        observed_vol = {
            "max_swing": max_swing,
            "avg_change": avg_change,
            "max_consec_change": max_consec_change,
            "std_dev": std_dev,
            "sigma_per_step": sigma_per_step,
            "step_interval_s": sample_interval_s,
            "steps_per_min": steps_per_min,
            "samples": n,
            "span_s": total_span_s,
        }

    if spot_price_info and spot_price_info.get("price_diff") is not None and total_seconds > 0 and observed_vol:
        abs_diff = abs(spot_price_info["price_diff"])
        diff = spot_price_info["price_diff"]
        mins_left = total_seconds / 60

        # Project expected swing using random walk: sigma * sqrt(steps_remaining)
        steps_remaining = mins_left * observed_vol["steps_per_min"]
        expected_swing = observed_vol["sigma_per_step"] * math.sqrt(steps_remaining) if steps_remaining > 0 else 0

        # How many "expected swings" is the current move?
        # >2x = very strong, 1-2x = solid, 0.5-1x = moderate, <0.5x = noise
        strength = abs_diff / expected_swing if expected_swing > 0 else 0

        parts.append(f"\n## Dynamic Risk Assessment (based on observed volatility)")
        parts.append(f"- **Current position:** ${diff:+,.2f} from reference ({'ABOVE' if diff > 0 else 'BELOW'} → favors {'YES' if diff > 0 else 'NO'})")
        parts.append(f"- **Observed sigma per {observed_vol['step_interval_s']:.0f}s step:** ${observed_vol['sigma_per_step']:,.2f}")
        parts.append(f"- **Steps remaining (~{observed_vol['step_interval_s']:.0f}s each):** {steps_remaining:.0f}")
        parts.append(f"- **Expected random-walk swing in {mins_left:.1f}min:** ~${expected_swing:,.0f}")
        parts.append(f"- **Signal strength:** {strength:.1f}x expected swing")

        parts.append(f"\n### Observed Volatility Detail")
        parts.append(f"- **Max swing (high-low) in window:** ${observed_vol['max_swing']:,.2f}")
        parts.append(f"- **Avg change between consecutive samples:** ${observed_vol['avg_change']:,.2f}")
        parts.append(f"- **Max single-sample jump:** ${observed_vol['max_consec_change']:,.2f}")
        parts.append(f"- **Std deviation of prices:** ${observed_vol['std_dev']:,.2f}")

        if strength >= 2.0:
            parts.append(
                f"- **CONCLUSION: STRONG position.** The price is {strength:.1f}x the expected swing away from "
                f"the reference. Reversal is very unlikely in {mins_left:.1f}min. "
                f"{'BUY_YES is strongly supported.' if diff > 0 else 'BUY_NO is strongly supported.'}"
            )
        elif strength >= 1.0:
            parts.append(
                f"- **CONCLUSION: SOLID position.** The price is {strength:.1f}x the expected swing away from "
                f"the reference. Reversal is possible but unlikely. "
                f"{'BUY_YES is supported.' if diff > 0 else 'BUY_NO is supported.'}"
            )
        elif strength >= 0.5:
            parts.append(
                f"- **CONCLUSION: WEAK position.** The price is only {strength:.1f}x the expected swing. "
                f"Reversal is plausible. Trade with caution or SKIP."
            )
        else:
            parts.append(
                f"- **CONCLUSION: NOISE.** The price is only {strength:.1f}x the expected swing. "
                f"This is within normal random walk range. Essentially a coin flip. "
                f"Prefer SKIP or trade the CURRENT side with low confidence."
            )

    # Enriched market signals from Gamma
    if market_signals:
        parts.append("\n## Market Signals")
        if market_signals.get("one_hour_change"):
            parts.append(f"- 1h price change: {market_signals['one_hour_change']:+.4f}")
        if market_signals.get("one_day_change"):
            parts.append(f"- 24h price change: {market_signals['one_day_change']:+.4f}")
        if market_signals.get("volume_24h"):
            parts.append(f"- 24h volume: ${market_signals['volume_24h']:,.0f}")
        if market_signals.get("spread"):
            parts.append(f"- Spread: {market_signals['spread']:.4f}")
        if market_signals.get("last_trade_price"):
            parts.append(f"- Last trade price: {market_signals['last_trade_price']:.4f}")

    # Recent price history
    if price_history:
        parts.append("\n## Recent Price History (oldest → newest)")
        # Show up to last 12 points to avoid overwhelming the prompt
        recent = price_history[-12:]
        prices_str = ", ".join(f"{p.get('p', '?')}" for p in recent)
        parts.append(f"Prices: [{prices_str}]")
        # Show trend summary
        if len(recent) >= 2:
            first_p = float(recent[0].get("p", 0.5))
            last_p = float(recent[-1].get("p", 0.5))
            delta = last_p - first_p
            direction = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
            parts.append(f"Trend: {direction} ({delta:+.4f} over {len(recent)} points)")

    # Order book summary
    if orderbook_summary:
        parts.append("\n## Order Book Summary")
        parts.append(f"- Best bid: {orderbook_summary.get('best_bid', '?')}")
        parts.append(f"- Best ask: {orderbook_summary.get('best_ask', '?')}")
        parts.append(f"- Spread: {orderbook_summary.get('spread', '?')}")
        parts.append(f"- Bid depth: {orderbook_summary.get('bid_depth', '?')}")
        parts.append(f"- Ask depth: {orderbook_summary.get('ask_depth', '?')}")
        ratio = orderbook_summary.get("bid_ask_ratio")
        if ratio is not None:
            pressure = "BUYING pressure" if ratio > 1 else "SELLING pressure" if ratio < 1 else "NEUTRAL"
            parts.append(f"- Bid/Ask ratio: {ratio:.3f} ({pressure})")

    # Past evaluation history (agent memory across cycles)
    if past_evaluations:
        parts.append(f"\n## Your Previous Evaluations ({len(past_evaluations)} most recent)")
        parts.append("Use these to track how the market and your assessment have evolved:")
        for i, ev in enumerate(past_evaluations, 1):
            parts.append(
                f"  {i}. [{ev.get('timestamp', '?')}] "
                f"YES=${ev.get('yes_price', '?'):.2f} NO=${ev.get('no_price', '?'):.2f} → "
                f"{ev.get('recommendation', '?')} "
                f"(conf={ev.get('confidence', '?'):.0%}, "
                f"est_prob={ev.get('estimated_probability', '?'):.0%}) "
                f"| {ev.get('reasoning', '')}"
            )

    if order_history:
        parts.append(f"\n## Orders Placed This Session ({len(order_history)} orders)")
        parts.append("You have already placed these orders on this market:")
        for i, order in enumerate(order_history, 1):
            parts.append(
                f"  {i}. [{order.get('timestamp', '?')}] "
                f"{order.get('action', '?')} "
                f"size={order.get('size', '?')} "
                f"price={order.get('price', '?')}"
            )

    # Accumulated spot price history from PriceTracker (30s intervals)
    if price_tracker_data:
        parts.append(f"\n## Accumulated Spot Price History ({len(price_tracker_data)} snapshots, 30s intervals)")
        parts.append("Chronological price samples from Binance:")
        for snap in price_tracker_data:
            diff_str = ""
            if snap.get("diff") is not None:
                diff_str = f" | diff=${snap['diff']:+,.2f} ({snap.get('diff_pct', 0):+.3f}%)"
            parts.append(
                f"  [{snap.get('timestamp', '?')}] "
                f"${snap['spot_price']:,.2f}"
                f"{diff_str}"
            )
        # Quick trend summary
        if len(price_tracker_data) >= 2:
            first_s = price_tracker_data[0]["spot_price"]
            last_s = price_tracker_data[-1]["spot_price"]
            delta = last_s - first_s
            pct = (delta / first_s * 100) if first_s else 0
            direction = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
            parts.append(f"  **Trend:** {direction} ${delta:+,.2f} ({pct:+.3f}%) over {len(price_tracker_data)} samples")

    # AI reasoning chain from prior analysis cycles
    if reasoning_chain:
        parts.append(f"\n## AI Reasoning Chain ({len(reasoning_chain)} prior analyses)")
        parts.append("Your previous analyses of this market (use to build on your reasoning):")
        for i, r in enumerate(reasoning_chain, 1):
            spot_str = f" spot=${r['spot_price']:,.2f}" if r.get("spot_price") else ""
            parts.append(
                f"  {i}. [{r.get('timestamp', '?')}] "
                f"{r.get('recommendation', '?')} "
                f"(conf={r.get('confidence', 0):.0%}){spot_str}"
                f"\n     Reasoning: {r.get('reasoning', '')}"
            )

    if positions:
        parts.append("\n## Current Portfolio Positions")
        for pos in positions:
            token = pos.get("outcome", "?")
            size = pos.get("size", 0)
            parts.append(f"- {token}: {size} shares")

    parts.append(
        "\nAnalyze this market using ALL the data above — price trends, order book pressure, "
        "and market signals. Identify any edge and make a concrete recommendation. "
        "Respond with JSON only."
    )

    return "\n".join(p for p in parts if p)
