"""Configuration via pydantic-settings, loaded from .env."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
    )

    # Wallet
    private_key: str = ""
    wallet_address: str = ""
    proxy_address: str = ""  # Polymarket proxy wallet (funder) address

    # LLM
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "https://opencode.ai/zen/v1"
    llm_model: str = "opencode/claude-sonnet-4-5"

    # Trading limits (hard caps)
    max_bet_usdc: float = 5.0
    max_portfolio_usdc: float = 500.0

    # Market filters
    min_liquidity: float = 1000.0
    min_volume: float = 5000.0

    # Agent timing (seconds)
    price_sample_interval: int = 5  # Sample prices every 5s for better volatility data
    analysis_interval: int = 20  # Analyze every 20s (includes LLM call)
    trade_window_secs: int = 180  # Only trade in last 3 min — accumulate data before that

    # API endpoints
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    data_api_url: str = "https://data-api.polymarket.com"

    # Polymarket chain
    chain_id: int = 137  # Polygon mainnet


settings = Settings()
