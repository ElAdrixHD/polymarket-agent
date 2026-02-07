"""Gamma API client for market discovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from polymarket_agent.config import settings

BASE = settings.gamma_api_url

# Known tag slugs on Polymarket for faster discovery
# Maps user-facing keywords → Gamma API tag_slug values
TAG_SLUGS: dict[str, list[str]] = {
    "crypto": ["crypto", "crypto-prices"],
    "bitcoin": ["bitcoin", "crypto-prices"],
    "btc": ["bitcoin", "crypto-prices"],
    "ethereum": ["crypto", "crypto-prices"],
    "eth": ["crypto", "crypto-prices"],
    "solana": ["crypto", "crypto-prices"],
    "sol": ["crypto", "crypto-prices"],
    "xrp": ["crypto", "crypto-prices"],
    "up or down": ["up-or-down", "15M", "5M", "1H"],
    "updown": ["up-or-down", "15M", "5M", "1H"],
    "5min": ["5M", "up-or-down"],
    "5m": ["5M", "up-or-down"],
    "15min": ["15M", "up-or-down"],
    "15m": ["15M", "up-or-down"],
    "1h": ["1H", "up-or-down"],
    "1hour": ["1H", "up-or-down"],
    "politics": ["politics"],
    "trump": ["politics", "us-elections"],
    "election": ["politics", "us-elections"],
    "sports": ["sports"],
    "nfl": ["sports"],
    "nba": ["sports"],
    "soccer": ["sports"],
    "ai": ["ai"],
    "tech": ["science-tech"],
    "science": ["science-tech"],
    "pop culture": ["pop-culture"],
}


async def get_active_events(
    limit: int = 20,
    offset: int = 0,
    tag_id: int | None = None,
) -> list[dict[str, Any]]:
    """List active events."""
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "active": True,
        "closed": False,
    }
    if tag_id is not None:
        params["tag_id"] = tag_id
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE}/events", params=params)
        resp.raise_for_status()
        return resp.json()


async def get_event(slug: str) -> dict[str, Any]:
    """Get event detail by slug."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE}/events", params={"slug": slug})
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"Event not found: {slug}")
        return data[0]


async def get_active_markets(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """List active markets."""
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "active": True,
        "closed": False,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


async def get_market(slug: str) -> dict[str, Any]:
    """Get market detail by slug (includes token IDs)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE}/markets", params={"slug": slug})
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"Market not found: {slug}")
        return data[0]


async def search_markets(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search markets by text."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE}/markets",
            params={"_q": query, "limit": limit, "active": True, "closed": False},
        )
        resp.raise_for_status()
        return resp.json()


async def _search_events_by_tag(tag_slug: str, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch events by tag slug and extract their markets."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BASE}/events",
            params={"tag_slug": tag_slug, "limit": limit, "active": True, "closed": False},
        )
        resp.raise_for_status()
        events = resp.json()

    markets: list[dict[str, Any]] = []
    for event in events:
        for m in event.get("markets") or []:
            # Carry event title for context
            m.setdefault("_event_title", event.get("title", ""))
            markets.append(m)
    return markets


def _text_matches(market: dict[str, Any], keywords: list[str]) -> bool:
    """Check if any keyword appears in the market question or event title."""
    question = (market.get("question") or "").lower()
    event_title = (market.get("_event_title") or "").lower()
    slug = (market.get("slug") or "").lower()
    text = f"{question} {event_title} {slug}"
    return any(kw.lower() in text for kw in keywords)


# Timeframe keywords that indicate short-term up/down markets
_SHORT_TERM_KEYWORDS: set[str] = {
    "5min", "5m", "15min", "15m", "1h", "1hour",
    "up or down", "updown",
}


def _is_short_term_query(terms: list[str]) -> bool:
    """Detect if the query targets short-term (5m/15m/1h) up/down markets."""
    return any(t.lower().strip() in _SHORT_TERM_KEYWORDS for t in terms)


def _filter_short_term_markets(
    markets: list[dict[str, Any]],
    terms: list[str],
) -> list[dict[str, Any]]:
    """For short-term queries, keep only up/down markets and pick the live one per asset.

    1. Filter to only markets whose slug matches the updown pattern
       (e.g. btc-updown-15m-...) or whose question contains "up or down".
    2. If the user specified an asset (btc, eth, sol, xrp), filter to that asset only.
    3. Group by asset+timeframe prefix and keep only the nearest-expiring (= live now).
    """
    import re

    now = datetime.now(timezone.utc)

    # Determine which asset(s) the user asked about
    asset_keywords = {"btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp"}
    requested_assets: set[str] = set()
    for t in terms:
        low = t.lower().strip()
        if low in asset_keywords:
            # Normalize to slug prefix form
            asset_map = {
                "btc": "btc", "bitcoin": "btc",
                "eth": "eth", "ethereum": "eth",
                "sol": "sol", "solana": "sol",
                "xrp": "xrp",
            }
            if low in asset_map:
                requested_assets.add(asset_map[low])

    # Determine which timeframe(s) the user asked about
    timeframe_keywords = {
        "5min": "5m", "5m": "5m", "5": "5m",
        "15min": "15m", "15m": "15m", "15": "15m",
        "1h": "1h", "1hour": "1h",
    }
    requested_timeframes: set[str] = set()
    for t in terms:
        low = t.lower().strip()
        if low in timeframe_keywords:
            requested_timeframes.add(timeframe_keywords[low])

    # Step 1: Filter to only updown markets
    updown_pattern = re.compile(r"^([a-z]+)-updown-(\d+[mh])-")
    updown_markets: list[tuple[str, str, dict[str, Any]]] = []  # (asset, timeframe, market)

    for m in markets:
        slug = (m.get("slug") or "").lower()
        match = updown_pattern.match(slug)
        if not match:
            # Also check question text as fallback
            question = (m.get("question") or "").lower()
            if "up or down" not in question:
                continue
            # Try to extract asset/timeframe from question
            # e.g. "Bitcoin Up or Down - February 6, 9:15AM-9:30AM ET"
            match = updown_pattern.match(slug)
            if not match:
                continue
        asset = match.group(1)
        timeframe = match.group(2)

        # Step 2: Filter by requested asset
        if requested_assets and asset not in requested_assets:
            continue

        # Filter by requested timeframe
        if requested_timeframes and timeframe not in requested_timeframes:
            continue

        updown_markets.append((asset, timeframe, m))

    # Step 3: Group by asset+timeframe, keep nearest-expiring
    buckets: dict[str, list[dict[str, Any]]] = {}
    for asset, timeframe, m in updown_markets:
        key = f"{asset}-{timeframe}"
        buckets.setdefault(key, []).append(m)

    # Skip markets closing within this buffer — they're about to end,
    # pick the next window instead.
    min_remaining = timedelta(minutes=2)

    result: list[dict[str, Any]] = []
    for key, group in buckets.items():
        group.sort(key=lambda m: _parse_end_date(m) or datetime.max.replace(tzinfo=timezone.utc))
        for m in group:
            end_dt = _parse_end_date(m)
            if end_dt and end_dt > now + min_remaining:
                result.append(m)
                break
        else:
            if group:
                result.append(group[0])

    return result


def passes_filters(market: dict[str, Any]) -> bool:
    """Check if a market passes minimum liquidity/volume filters."""
    volume = float(market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidity", 0) or 0)
    return volume >= settings.min_volume and liquidity >= settings.min_liquidity


def _parse_end_date(market: dict[str, Any]) -> datetime | None:
    """Parse end date from a market dict."""
    end_date = market.get("endDate") or market.get("end_date_iso")
    if not end_date:
        return None
    try:
        end_str = str(end_date).replace("Z", "+00:00")
        end_dt = datetime.fromisoformat(end_str)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        return end_dt
    except (ValueError, TypeError):
        return None


def is_still_active(market: dict[str, Any]) -> bool:
    """Check if a market hasn't expired yet."""
    end_dt = _parse_end_date(market)
    if end_dt is None:
        return True  # No end date = assume still active
    return end_dt > datetime.now(timezone.utc)


def expires_within_hours(market: dict[str, Any], max_hours: float) -> bool:
    """Check if a market expires within the given number of hours from now."""
    end_dt = _parse_end_date(market)
    if end_dt is None:
        return False
    now = datetime.now(timezone.utc)
    diff_hours = (end_dt - now).total_seconds() / 3600
    return 0 < diff_hours <= max_hours


# Canonical asset keywords for auto-expansion
_ASSET_KEYWORDS: set[str] = {"btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp"}
_ALL_CRYPTO_ASSETS: list[str] = ["btc", "eth", "sol", "xrp"]


def _should_auto_expand(terms: list[str]) -> bool:
    """Return True if terms contain a timeframe keyword but NO asset keyword."""
    has_timeframe = any(t.lower().strip() in _SHORT_TERM_KEYWORDS for t in terms)
    has_asset = any(t.lower().strip() in _ASSET_KEYWORDS for t in terms)
    return has_timeframe and not has_asset


async def search_and_filter(
    terms: list[str],
    max_hours: float | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Search markets by terms using tag-based discovery + text filtering.

    Strategy:
    1. Map search terms to known tag slugs for broad discovery
    2. Text-filter results to match the actual keywords
    3. Optionally filter by expiration time
    4. Deduplicate by market ID
    5. If terms have a timeframe but no asset, auto-expand to all crypto assets
    """
    # Auto-expand: if user said "15min" with no asset, search all assets
    if _should_auto_expand(terms):
        terms = terms + _ALL_CRYPTO_ASSETS

    seen_ids: set[str] = set()
    results: list[dict[str, Any]] = []

    # Collect tag slugs to search
    tag_slugs_to_search: set[str] = set()
    for term in terms:
        key = term.lower().strip()
        if key in TAG_SLUGS:
            tag_slugs_to_search.update(TAG_SLUGS[key])

    def _should_include(m: dict[str, Any]) -> bool:
        mid = m.get("id") or m.get("conditionId") or m.get("slug", "")
        if mid in seen_ids:
            return False
        seen_ids.add(mid)
        if not is_still_active(m):
            return False
        if not _text_matches(m, terms):
            return False
        if max_hours is not None and not expires_within_hours(m, max_hours):
            return False
        return True

    # Search via tag-based event discovery
    if tag_slugs_to_search:
        for tag_slug in tag_slugs_to_search:
            markets = await _search_events_by_tag(tag_slug, limit=limit)
            for m in markets:
                if _should_include(m):
                    results.append(m)

    # Fallback: also search via the markets endpoint with each term
    for term in terms:
        markets = await search_markets(term, limit=limit)
        for m in markets:
            if _should_include(m):
                results.append(m)

    # For short-term queries, filter to only the live up/down market per asset
    if _is_short_term_query(terms):
        filtered = _filter_short_term_markets(results, terms)
        if filtered:
            results = filtered

    # Sort by volume descending
    results.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
    return results


def extract_market_signals(market: dict[str, Any]) -> dict[str, Any]:
    """Extract enriched trading signals from a Gamma market response."""
    def _f(key: str, default: float = 0.0) -> float:
        val = market.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    best_bid = _f("bestBid")
    best_ask = _f("bestAsk")
    spread = _f("spread", best_ask - best_bid if best_ask and best_bid else 0.0)

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": round(spread, 4),
        "last_trade_price": _f("lastTradePrice"),
        "one_hour_change": _f("oneHourPriceChange"),
        "one_day_change": _f("oneDayPriceChange"),
        "volume_24h": _f("volume24hr"),
        "outcome_prices": market.get("outcomePrices", market.get("outcome_prices", "")),
    }
