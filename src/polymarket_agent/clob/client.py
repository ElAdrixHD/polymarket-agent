"""CLOB client wrapper for trading operations."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams, OrderArgs, OrderType

from polymarket_agent.config import settings

logger = logging.getLogger(__name__)

_client: ClobClient | None = None
_initialized: bool = False


def _get_client() -> ClobClient:
    global _client
    if _client is None:
        _client = ClobClient(
            settings.clob_api_url,
            key=settings.private_key,
            chain_id=settings.chain_id,
            signature_type=2,  # POLY_PROXY (Polymarket UI wallet)
            funder=settings.proxy_address or None,
        )
    return _client


def init() -> ClobClient:
    """Initialize and return the CLOB client with API creds."""
    global _initialized
    client = _get_client()
    if not _initialized:
        client.set_api_creds(client.create_or_derive_api_creds())
        # Refresh server-side allowance tracking
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            client.update_balance_allowance(params)
            logger.info("Balance/allowance refreshed on CLOB server")
        except Exception as exc:
            logger.warning("Failed to refresh balance/allowance: %s", exc)
        _initialized = True
    return client


def _get_initialized_client() -> ClobClient:
    """Get a client that has API creds set up (needed for orderbook etc)."""
    global _initialized
    client = _get_client()
    if not _initialized and settings.private_key:
        try:
            client.set_api_creds(client.create_or_derive_api_creds())
            _initialized = True
        except Exception as exc:
            logger.warning("Failed to initialize CLOB API creds: %s", exc)
    return client


def get_balance() -> float:
    """Get available USDC balance from the CLOB."""
    try:
        client = init()
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=2,  # Match proxy wallet type
        )
        resp = client.get_balance_allowance(params)
        raw = float(resp.get("balance", 0) or 0)
        # API returns raw USDC units (6 decimals)
        balance = raw / 1e6 if raw > 1_000 else raw
        logger.info("CLOB balance response: raw=%s parsed=$%.2f", raw, balance)
        return balance
    except Exception as exc:
        logger.warning("Failed to get balance: %s", exc)
        return 0.0


def get_price(token_id: str, side: str = "BUY") -> dict[str, Any]:
    """Get current price for a token (requires side: BUY or SELL)."""
    client = _get_client()
    return client.get_price(token_id, side)


def get_midpoint(token_id: str) -> float:
    """Get the midpoint price for a token. Returns 0.0 on failure."""
    try:
        client = _get_client()
        resp = client.get_midpoint(token_id)
        return float(resp.get("mid", 0) or 0)
    except Exception as exc:
        logger.warning("Failed to get midpoint for %s: %s", token_id[:20], exc)
        return 0.0


def get_last_trade_price(token_id: str) -> float:
    """Get the last trade price for a token. Returns 0.0 on failure."""
    try:
        client = _get_client()
        resp = client.get_last_trade_price(token_id)
        return float(resp.get("price", 0) or 0)
    except Exception as exc:
        logger.warning("Failed to get last trade price for %s: %s", token_id[:20], exc)
        return 0.0


def get_orderbook(token_id: str) -> dict[str, Any]:
    """Get the order book for a token."""
    client = _get_initialized_client()
    return client.get_order_book(token_id)


def buy(token_id: str, price: float, size: float) -> dict[str, Any]:
    """Place a GTC buy order."""
    client = _get_client()
    order_args = OrderArgs(
        price=price,
        size=size,
        side="BUY",
        token_id=token_id,
    )
    signed = client.create_order(order_args)
    return client.post_order(signed, OrderType.GTC)


def sell(token_id: str, price: float, size: float) -> dict[str, Any]:
    """Place a GTC sell order."""
    client = _get_client()
    order_args = OrderArgs(
        price=price,
        size=size,
        side="SELL",
        token_id=token_id,
    )
    signed = client.create_order(order_args)
    return client.post_order(signed, OrderType.GTC)


def market_buy(token_id: str, amount: float, max_price: float = 0.99) -> dict[str, Any]:
    """Place a FOK market buy order.

    *max_price* is the ceiling price — the order will only fill if the
    available price is at or below this value.  Defaults to 0.99 (max).
    """
    client = _get_client()
    # Clamp to valid Polymarket range
    price = max(0.01, min(max_price, 0.99))
    order_args = OrderArgs(
        price=price,
        size=amount,
        side="BUY",
        token_id=token_id,
    )
    signed = client.create_order(order_args)
    return client.post_order(signed, OrderType.FOK)


def get_open_orders() -> list[dict[str, Any]]:
    """Get all open orders."""
    client = _get_client()
    return client.get_orders()


def cancel_all() -> list[Any]:
    """Cancel all open orders."""
    client = _get_client()
    return client.cancel_all()


async def get_price_history(
    token_id: str,
    interval: str = "1h",
    fidelity: int = 5,
) -> list[dict[str, Any]]:
    """Fetch recent price history from the CLOB prices-history endpoint.

    Returns a list of ``{t, p}`` dicts (timestamp, price).
    """
    url = f"{settings.clob_api_url}/prices-history"
    params = {"market": token_id, "interval": interval, "fidelity": fidelity}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json().get("history", [])
    except Exception as exc:
        logger.warning("Failed to fetch price history: %s", exc)
        return []


def check_orderbook_liquidity(token_id: str, side: str, size: float, max_price: float) -> bool:
    """Check if the order book has enough liquidity to fill a FOK order.

    For BUY orders, sums ask sizes at or below *max_price*.
    Returns True if available liquidity >= requested *size*.
    """
    try:
        book = get_orderbook(token_id)
        if hasattr(book, "asks"):
            asks = book.asks or []
            bids = book.bids or []
        else:
            asks = book.get("asks", [])
            bids = book.get("bids", [])

        def _val(entry: Any, key: str) -> float:
            if hasattr(entry, key):
                return float(getattr(entry, key, 0))
            return float(entry.get(key, 0)) if isinstance(entry, dict) else 0.0

        if side.upper() == "BUY":
            available = sum(
                _val(o, "size") for o in asks if _val(o, "price") <= max_price
            )
        else:
            available = sum(
                _val(o, "size") for o in bids if _val(o, "price") >= max_price
            )

        logger.info(
            "Liquidity check: side=%s size=%.2f max_price=%.2f available=%.2f sufficient=%s",
            side, size, max_price, available, available >= size,
        )
        return available >= size
    except Exception as exc:
        logger.warning("Liquidity check failed: %s — proceeding with order", exc)
        return True  # On error, let the FOK decide


def get_orderbook_summary(token_id: str) -> dict[str, Any]:
    """Get a compact summary of the current order book.

    Returns bid_depth, ask_depth, spread, midpoint, and bid/ask ratio.
    """
    try:
        book = get_orderbook(token_id)
        # Handle both dict and OrderBookSummary object
        if hasattr(book, "bids"):
            bids = book.bids or []
            asks = book.asks or []
        else:
            bids = book.get("bids", [])
            asks = book.get("asks", [])

        def _get(entry: Any, key: str, default: float = 0.0) -> float:
            if hasattr(entry, key):
                return float(getattr(entry, key, default))
            return float(entry.get(key, default)) if isinstance(entry, dict) else default

        bid_depth = sum(_get(o, "size") for o in bids)
        ask_depth = sum(_get(o, "size") for o in asks)

        best_bid = _get(bids[0], "price") if bids else 0.0
        best_ask = _get(asks[0], "price") if asks else 1.0

        spread = best_ask - best_bid
        midpoint = (best_bid + best_ask) / 2 if (best_bid + best_ask) else 0.5
        ratio = bid_depth / ask_depth if ask_depth else float("inf")

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 4),
            "midpoint": round(midpoint, 4),
            "bid_depth": round(bid_depth, 2),
            "ask_depth": round(ask_depth, 2),
            "bid_ask_ratio": round(ratio, 3),
        }
    except Exception as exc:
        logger.warning("Failed to get orderbook summary: %s", exc)
        return {}
