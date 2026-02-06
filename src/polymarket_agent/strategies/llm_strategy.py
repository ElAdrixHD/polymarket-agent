"""LLM-powered trading strategy."""

from __future__ import annotations

from typing import Any

from polymarket_agent.analyst.llm import analyze_market
from polymarket_agent.config import settings
from polymarket_agent.strategies.base import TradeSignal


class LLMStrategy:
    """Evaluate markets using LLM analysis with risk filters."""

    def __init__(self, risk_tolerance: str | None = None) -> None:
        self.risk_tolerance = risk_tolerance

    async def evaluate(
        self,
        market: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> TradeSignal | None:
        analysis = await analyze_market(market, positions, risk_tolerance=self.risk_tolerance)

        if analysis.recommendation == "SKIP":
            return None

        if analysis.confidence < settings.min_confidence:
            return None

        # Determine token ID based on recommendation
        tokens = market.get("tokens", [])
        token_id = ""
        for tok in tokens:
            outcome = tok.get("outcome", "").upper()
            if analysis.recommendation == "BUY_YES" and outcome == "YES":
                token_id = tok.get("token_id", "")
            elif analysis.recommendation == "BUY_NO" and outcome == "NO":
                token_id = tok.get("token_id", "")

        if not token_id:
            return None

        # Determine size (capped by max bet)
        size = min(
            analysis.suggested_size or settings.max_bet_usdc,
            settings.max_bet_usdc,
        )

        # Determine price
        price = analysis.suggested_price
        if price is None:
            for tok in tokens:
                if tok.get("token_id") == token_id:
                    price = float(tok.get("price", 0.5))
                    break
            else:
                price = 0.5

        # Check portfolio exposure
        current_exposure = sum(
            float(p.get("size", 0) or 0) * float(p.get("currentPrice", 0) or 0)
            for p in positions
        )
        if current_exposure + (size * price) > settings.max_portfolio_usdc:
            size = max(0, (settings.max_portfolio_usdc - current_exposure) / price)
            if size < 1:
                return None

        return TradeSignal(
            action=analysis.recommendation,
            token_id=token_id,
            price=price,
            size=size,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
        )
