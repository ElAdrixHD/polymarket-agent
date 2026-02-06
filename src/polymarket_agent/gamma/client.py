"""Gamma API client for market discovery."""

from __future__ import annotations

from datetime import datetime, timezone
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
    """
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

    # Sort by volume descending
    results.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
    return results
