"""Prompt templates for LLM market analysis."""

from datetime import datetime, timezone

QUERY_SYSTEM_PROMPT = """\
You are a query interpreter for a Polymarket prediction market agent.
The user will ask you to find or analyze specific markets in natural language.

Extract the search parameters from the user's request and respond with valid JSON:
{
  "search_terms": ["keyword1", "keyword2"],
  "max_hours": number or null,
  "action": "search" | "analyze",
  "risk_tolerance": "string" or null
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
- risk_tolerance: if the user mentions risk preference, extract it. null otherwise.

Polymarket has these short-term "Up or Down" markets for crypto (BTC, ETH, SOL, XRP) in 5min, 15min, and 1h intervals. \
Their slugs look like: btc-updown-15m-{timestamp}, eth-updown-5m-{timestamp}, etc.

Examples:
- "show me btc bets expiring in 1h" → {"search_terms": ["bitcoin", "btc"], "max_hours": 1, "action": "search", "risk_tolerance": null}
- "analyze btc 15min" → {"search_terms": ["bitcoin", "btc", "up or down", "15"], "max_hours": null, "action": "analyze", "risk_tolerance": null}
- "analyze crypto markets closing in 15min aggressively" → {"search_terms": ["crypto", "bitcoin", "ethereum"], "max_hours": 0.25, "action": "analyze", "risk_tolerance": "aggressive"}
- "what political bets are there?" → {"search_terms": ["politics", "trump", "election"], "max_hours": null, "action": "search", "risk_tolerance": null}
- "btc 1h markets" → {"search_terms": ["bitcoin", "btc", "up or down", "1h"], "max_hours": null, "action": "search", "risk_tolerance": null}
- "eth 5min" → {"search_terms": ["ethereum", "eth", "up or down", "5"], "max_hours": null, "action": "search", "risk_tolerance": null}

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
  - The magnitude of the difference relative to time remaining determines confidence.
  - **This is real external data, not market opinion. Trust it over order book sentiment.**
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

## Short-Term Markets (Up or Down)
These are binary markets that resolve within minutes or hours. Key considerations:
- **Time decay matters**: as resolution approaches, prices should converge to true probability fast.
- **SKIP = guaranteed loss of opportunity**: these markets expire quickly, so being in the market \
with even a small edge is better than missing it entirely.
- **Momentum is king**: in short timeframes, recent price trends and order book pressure are the \
strongest signals. Fundamentals matter less.
- **NEVER fight the trend**: if price is dropping sharply (e.g. YES went from 0.50 → 0.20), that is \
strong bearish momentum — do NOT buy YES thinking it's "undervalued" or "cheap". A falling price means \
the market is repricing downward. Follow the momentum direction, not "value".
- **A large price drop is NOT a buying opportunity**: if YES dropped 50%+ in minutes, that's a crash, \
not a discount. The correct trade is BUY_NO (or SKIP), never BUY_YES against the trend.
- **Use price history direction**: look at the TREND in price history, not individual price levels. \
If the last N points show consistent decline, trade WITH that direction.
- **Lower your edge threshold**: for markets expiring in < 30 minutes, an edge of 2 cents is actionable.

## Decision Framework
1. Look at the price trend — is there clear momentum in one direction?
2. Check order book pressure — does it confirm or contradict the trend?
3. Estimate the true probability based on all available signals
4. Compare to current price to find edges >= 3 cents (2 cents for short-term markets)
5. Only SKIP if you truly cannot form an opinion or the edge is < 2 cents

Always respond with ONLY valid JSON (no markdown, no extra text) matching this schema:
{
  "recommendation": "BUY_YES" | "BUY_NO" | "SKIP",
  "confidence": 0.0 to 1.0,
  "estimated_probability": 0.0 to 1.0,
  "reasoning": "short explanation, max 2 sentences",
  "suggested_price": 0.0 to 1.0 or null,
  "suggested_size": number or null
}
"""

RISK_INSTRUCTIONS = {
    "_default": (
        "\n## Risk Profile\n"
        "The user has described their risk tolerance as: \"{risk_tolerance}\"\n\n"
        "Adapt your analysis to match this risk profile:\n"
        "- Adjust the minimum edge you require before recommending a trade. "
        "A more aggressive profile means you can recommend trades with smaller edges; "
        "a conservative profile means you need larger mispricings.\n"
        "- Adjust suggested_size relative to the user's tolerance. "
        "Aggressive → larger positions, conservative → smaller positions.\n"
        "- Adjust confidence thresholds. "
        "Conservative → only recommend when very confident, aggressive → act on weaker signals.\n"
        "- Always explain in your reasoning how the risk profile influenced your decision.\n"
    ),
}


def build_system_prompt(risk_tolerance: str) -> str:
    """Build the full system prompt including risk instructions."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    base = SYSTEM_PROMPT.replace("{current_time}", now)
    risk_block = RISK_INSTRUCTIONS["_default"].format(risk_tolerance=risk_tolerance)
    return base + risk_block


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

    # Dynamic risk guidance based on time remaining + price movement
    # Uses dollar-based volatility thresholds calibrated per asset.
    # Typical BTC volatility: ~$50/min, ~$150/5min, ~$300/15min
    # Typical ETH volatility: ~$5/min, ~$15/5min, ~$30/15min
    _VOLATILITY_PER_MIN: dict[str, float] = {
        "BTC": 50.0, "ETH": 5.0, "SOL": 0.20, "XRP": 0.002,
    }

    if spot_price_info and spot_price_info.get("price_diff") is not None and total_seconds > 0:
        abs_diff = abs(spot_price_info["price_diff"])
        diff = spot_price_info["price_diff"]
        asset = spot_price_info["asset"]
        mins_left = total_seconds / 60

        # Expected volatility for the remaining time (scales with sqrt of time)
        vol_per_min = _VOLATILITY_PER_MIN.get(asset, 50.0)
        expected_swing = vol_per_min * (mins_left ** 0.5)

        # How many "expected swings" is the current move?
        # >2x = very strong, 1-2x = solid, 0.5-1x = moderate, <0.5x = noise
        strength = abs_diff / expected_swing if expected_swing > 0 else 0

        parts.append(f"\n## Dynamic Risk Assessment")
        parts.append(f"- **Price move:** ${diff:+,.2f} from reference")
        parts.append(f"- **Expected {asset} swing in {mins_left:.0f}min:** ~${expected_swing:,.0f}")
        parts.append(f"- **Signal strength:** {strength:.1f}x expected volatility")

        if strength >= 2.0:
            parts.append(
                f"- **RISK: LOW** — Move is {strength:.1f}x the expected swing. "
                f"Very strong signal. {asset} would need an extraordinary reversal "
                f"of ${abs_diff:,.0f} in {mins_left:.0f}min to flip. High confidence trade."
            )
        elif strength >= 1.0:
            parts.append(
                f"- **RISK: LOW-MODERATE** — Move is {strength:.1f}x the expected swing. "
                f"Solid signal. Reversal is possible but unlikely in {mins_left:.0f}min. "
                f"Trade with good confidence."
            )
        elif strength >= 0.5:
            parts.append(
                f"- **RISK: MODERATE** — Move is only {strength:.1f}x the expected swing. "
                f"The current direction could easily reverse. Trade with caution, smaller size."
            )
        else:
            parts.append(
                f"- **RISK: HIGH** — Move is only {strength:.1f}x the expected swing. "
                f"This is within normal noise for {asset}. Essentially a coin flip. "
                f"Prefer SKIP unless other signals are very strong."
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
