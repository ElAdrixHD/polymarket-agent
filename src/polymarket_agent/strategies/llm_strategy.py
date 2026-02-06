"""LLM-powered trading strategy."""

from __future__ import annotations

import logging
from typing import Any

from polymarket_agent.analyst.llm import analyze_market, extract_token_ids
from polymarket_agent.clob import client as clob
from polymarket_agent.config import settings
from polymarket_agent.strategies.base import TradeSignal

logger = logging.getLogger(__name__)


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

        logger.info(
            "LLM analysis for '%s': recommendation=%s confidence=%.2f "
            "est_prob=%.2f reasoning=%s",
            market.get("question", "?")[:60],
            analysis.recommendation,
            analysis.confidence,
            analysis.estimated_probability,
            analysis.reasoning,
        )

        if analysis.recommendation == "SKIP":
            return None

        if analysis.confidence < settings.min_confidence:
            logger.info(
                "Skipping: confidence %.2f < min %.2f",
                analysis.confidence,
                settings.min_confidence,
            )
            return None

        # Determine token ID based on recommendation
        token_ids = extract_token_ids(market)
        if analysis.recommendation == "BUY_YES":
            token_id = token_ids.get("YES", "")
        elif analysis.recommendation == "BUY_NO":
            token_id = token_ids.get("NO", "")
        else:
            token_id = ""

        if not token_id:
            logger.warning("No token_id found for %s", analysis.recommendation)
            return None

        # Determine size (capped by max bet, floored by Polymarket minimum of 5 shares)
        MIN_ORDER_SIZE = 5.0
        size = min(
            analysis.suggested_size or settings.max_bet_usdc,
            settings.max_bet_usdc,
        )

        # Determine price
        price = analysis.suggested_price
        if price is None:
            price = 0.5

        # Ensure minimum order size (5 shares on Polymarket)
        if size < MIN_ORDER_SIZE:
            size = MIN_ORDER_SIZE

        # Check available cash balance
        available = clob.get_balance()
        if available < size * price:
            size = available / price if price > 0 else 0
            logger.info("Capped size to %.1f based on $%.2f available", size, available)
            if size < MIN_ORDER_SIZE:
                logger.info("Skipping: size %.1f < minimum %s after balance cap", size, MIN_ORDER_SIZE)
                return None

        # Check portfolio exposure
        current_exposure = sum(
            float(p.get("size", 0) or 0) * float(p.get("currentPrice", 0) or 0)
            for p in positions
        )
        if current_exposure + (size * price) > settings.max_portfolio_usdc:
            size = max(0, (settings.max_portfolio_usdc - current_exposure) / price) if price > 0 else 0
            if size < MIN_ORDER_SIZE:
                return None

        return TradeSignal(
            action=analysis.recommendation,
            token_id=token_id,
            price=price,
            size=size,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
        )
