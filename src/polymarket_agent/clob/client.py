"""CLOB client wrapper for trading operations."""

from __future__ import annotations

from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from polymarket_agent.config import settings

_client: ClobClient | None = None


def _get_client() -> ClobClient:
    global _client
    if _client is None:
        _client = ClobClient(
            settings.clob_api_url,
            key=settings.private_key,
            chain_id=settings.chain_id,
            signature_type=0,  # EOA wallet
        )
    return _client


def init() -> ClobClient:
    """Initialize and return the CLOB client."""
    client = _get_client()
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def get_price(token_id: str) -> dict[str, Any]:
    """Get current price for a token."""
    client = _get_client()
    return client.get_price(token_id)


def get_orderbook(token_id: str) -> dict[str, Any]:
    """Get the order book for a token."""
    client = _get_client()
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


def market_buy(token_id: str, amount: float) -> dict[str, Any]:
    """Place a FOK market buy order."""
    client = _get_client()
    order_args = OrderArgs(
        price=1.0,
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
