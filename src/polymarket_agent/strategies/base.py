"""Base strategy interface."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class TradeSignal(BaseModel):
    action: str  # "BUY_YES", "BUY_NO"
    token_id: str
    price: float
    size: float
    confidence: float
    reasoning: str


class Strategy(Protocol):
    async def evaluate(
        self,
        market: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> TradeSignal | None: ...
