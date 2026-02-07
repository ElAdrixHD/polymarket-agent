"""Fetch live crypto spot prices from Binance public API (no auth required)."""

from __future__ import annotations

import logging
from collections import defaultdict
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


class PriceTracker:
    """Accumulates per-asset price snapshots over time for trend analysis."""

    def __init__(self) -> None:
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def record_snapshot(
        self,
        asset: str,
        spot_price: float,
        reference_price: float | None,
    ) -> dict[str, Any]:
        """Append a price snapshot for *asset* and return it."""
        diff = round(spot_price - reference_price, 4) if reference_price else None
        diff_pct = (
            round((spot_price - reference_price) / reference_price * 100, 4)
            if reference_price
            else None
        )
        snapshot = {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "spot_price": spot_price,
            "reference_price": reference_price,
            "diff": diff,
            "diff_pct": diff_pct,
        }
        self._history[asset.upper()].append(snapshot)
        return snapshot

    def get_history(self, asset: str) -> list[dict[str, Any]]:
        """Return the full list of snapshots for *asset*."""
        return list(self._history.get(asset.upper(), []))

    def get_summary(self, asset: str) -> str:
        """Return a formatted summary string for *asset*: trend, streaks, range."""
        history = self._history.get(asset.upper(), [])
        if not history:
            return f"{asset.upper()}: no data"

        spots = [s["spot_price"] for s in history]
        hi, lo = max(spots), min(spots)
        first, last = spots[0], spots[-1]
        total_change = last - first
        total_pct = (total_change / first * 100) if first else 0

        # Consecutive streak
        streak_dir = ""
        streak_count = 0
        for i in range(1, len(spots)):
            if spots[i] > spots[i - 1]:
                if streak_dir == "up":
                    streak_count += 1
                else:
                    streak_dir = "up"
                    streak_count = 1
            elif spots[i] < spots[i - 1]:
                if streak_dir == "down":
                    streak_count += 1
                else:
                    streak_dir = "down"
                    streak_count = 1
            # equal → streak continues as-is

        direction = "UP" if total_change > 0 else "DOWN" if total_change < 0 else "FLAT"
        lines = [
            f"{asset.upper()}: {len(history)} snapshots | {direction}",
            f"  Range: ${lo:,.2f} – ${hi:,.2f}",
            f"  Total change: ${total_change:+,.2f} ({total_pct:+.3f}%)",
        ]
        if streak_count:
            lines.append(f"  Current streak: {streak_count} consecutive {streak_dir}")
        return "\n".join(lines)

    def clear(self, asset: str) -> None:
        """Reset history for *asset*."""
        self._history.pop(asset.upper(), None)

    def clear_all(self) -> None:
        """Reset history for all assets."""
        self._history.clear()
