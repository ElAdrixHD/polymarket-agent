"""Prompt templates for LLM market analysis."""

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
You are an expert prediction market analyst. Your job is to evaluate Polymarket \
prediction markets and recommend whether to buy YES tokens, buy NO tokens, or skip.

You must be calibrated: if you think the true probability is 60%, and YES trades at \
50 cents, that's a BUY_YES. If YES trades at 70 cents, that's a BUY_NO (or SKIP).

Always respond with valid JSON matching this schema:
{
  "recommendation": "BUY_YES" | "BUY_NO" | "SKIP",
  "confidence": 0.0 to 1.0,
  "estimated_probability": 0.0 to 1.0,
  "reasoning": "string",
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
    base = SYSTEM_PROMPT
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
) -> str:
    """Build the user prompt for market analysis."""
    parts = [
        f"## Market Analysis Request",
        f"**Question:** {question}",
        f"**Description:** {description}" if description else "",
        f"**Current YES price:** ${yes_price:.2f}",
        f"**Current NO price:** ${no_price:.2f}",
        f"**Volume:** ${volume:,.0f}",
        f"**Liquidity:** ${liquidity:,.0f}",
        f"**End date:** {end_date}",
    ]

    if positions:
        parts.append("\n## Current Portfolio Positions")
        for pos in positions:
            token = pos.get("outcome", "?")
            size = pos.get("size", 0)
            parts.append(f"- {token}: {size} shares")

    parts.append(
        "\nAnalyze this market. Consider current events, base rates, and whether "
        "the price reflects the true probability. Respond with JSON only."
    )

    return "\n".join(p for p in parts if p)
