"""Data API client for positions and trade history."""

from __future__ import annotations

from typing import Any

import httpx

from polymarket_agent.config import settings

BASE = settings.data_api_url


async def get_positions(address: str | None = None) -> list[dict[str, Any]]:
    """Get current positions for an address."""
    addr = address or settings.wallet_address
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE}/positions", params={"user": addr})
        resp.raise_for_status()
        return resp.json()


async def get_trades(address: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Get trade history for an address."""
    addr = address or settings.wallet_address
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE}/trades",
            params={"user": addr, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()


async def get_portfolio_value(address: str | None = None) -> float:
    """Calculate total portfolio value from positions."""
    positions = await get_positions(address)
    total = 0.0
    for pos in positions:
        size = float(pos.get("size", 0) or 0)
        price = float(pos.get("currentPrice", 0) or 0)
        total += size * price
    return total
