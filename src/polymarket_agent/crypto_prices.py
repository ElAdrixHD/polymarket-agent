"""Fetch live crypto spot prices from Binance public API (no auth required)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Maps Polymarket asset slugs to Binance symbol pairs
_BINANCE_SYMBOLS: dict[str, str] = {
    "btc": "BTCUSDT",
    "bitcoin": "BTCUSDT",
    "eth": "ETHUSDT",
    "ethereum": "ETHUSDT",
    "sol": "SOLUSDT",
    "solana": "SOLUSDT",
    "xrp": "XRPUSDT",
}

BINANCE_API = "https://api.binance.com/api/v3"


async def get_spot_price(asset: str) -> float | None:
    """Get the current spot price for a crypto asset. Returns None on failure."""
    symbol = _BINANCE_SYMBOLS.get(asset.lower())
    if not symbol:
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{BINANCE_API}/ticker/price", params={"symbol": symbol})
            resp.raise_for_status()
            return float(resp.json()["price"])
    except Exception as exc:
        logger.warning("Failed to fetch %s spot price: %s", symbol, exc)
        return None


async def get_price_at_time(asset: str, timestamp_ms: int) -> float | None:
    """Get the closing price of a 1-minute candle at a specific timestamp.

    Uses Binance klines API to fetch the candle that contains the given
    timestamp, returning its close price as the reference price.
    """
    symbol = _BINANCE_SYMBOLS.get(asset.lower())
    if not symbol:
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{BINANCE_API}/klines",
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": timestamp_ms,
                    "limit": 1,
                },
            )
            resp.raise_for_status()
            klines = resp.json()
            if klines:
                # kline format: [open_time, open, high, low, close, ...]
                # Use the open price since that's closest to the start of the window
                return float(klines[0][1])
    except Exception as exc:
        logger.warning("Failed to fetch %s price at %s: %s", symbol, timestamp_ms, exc)
    return None


def extract_asset_from_slug(slug: str) -> str | None:
    """Extract the crypto asset name from an up/down market slug.

    e.g. 'btc-updown-15m-1770401700' → 'btc'
    """
    import re
    match = re.match(r"^([a-z]+)-updown-", slug or "")
    return match.group(1) if match else None


def parse_event_start_time(market: dict[str, Any]) -> int | None:
    """Extract the event start time as millisecond timestamp.

    Tries ``eventStartTime`` first, then falls back to parsing the slug
    timestamp (e.g. ``btc-updown-15m-1770401700``).
    """
    import re

    # Prefer the explicit field
    est = market.get("eventStartTime")
    if est:
        try:
            dt = datetime.fromisoformat(str(est).replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            pass

    # Fallback: parse unix timestamp from slug
    slug = market.get("slug", "")
    match = re.search(r"-(\d{10,})$", slug)
    if match:
        return int(match.group(1)) * 1000

    return None
