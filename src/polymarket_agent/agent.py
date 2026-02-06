"""Main autonomous agent loop."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.table import Table

from polymarket_agent.clob import client as clob
from polymarket_agent.config import settings
from polymarket_agent.data import client as data
from polymarket_agent.gamma import client as gamma
from polymarket_agent.strategies.llm_strategy import LLMStrategy

log = logging.getLogger(__name__)
console = Console()


async def scan_markets(limit: int = 20) -> list[dict]:
    """Scan and filter active markets."""
    markets = await gamma.get_active_markets(limit=limit)
    return [m for m in markets if gamma.passes_filters(m)]


def print_markets(markets: list[dict]) -> None:
    """Pretty-print markets to console."""
    table = Table(title="Active Markets")
    table.add_column("Question", style="cyan", max_width=60)
    table.add_column("Volume", justify="right", style="green")
    table.add_column("Liquidity", justify="right", style="yellow")
    table.add_column("Slug", style="dim")

    for m in markets:
        table.add_row(
            m.get("question", "?")[:60],
            f"${float(m.get('volume', 0) or 0):,.0f}",
            f"${float(m.get('liquidity', 0) or 0):,.0f}",
            m.get("slug", ""),
        )
    console.print(table)


async def run_once(strategy: LLMStrategy) -> None:
    """Run one iteration of the agent loop."""
    console.print("\n[bold]Scanning markets...[/bold]")
    markets = await scan_markets(limit=50)
    console.print(f"Found {len(markets)} markets passing filters")

    positions = []
    try:
        positions = await data.get_positions()
    except Exception as exc:
        log.warning("Could not fetch positions: %s", exc)

    for market in markets:
        question = market.get("question", "?")
        slug = market.get("slug", "")
        console.print(f"\n[bold blue]Evaluating:[/bold blue] {question}")

        try:
            signal = await strategy.evaluate(market, positions)
        except Exception as exc:
            log.error("Error evaluating %s: %s", slug, exc)
            continue

        if signal is None:
            console.print("  → SKIP")
            continue

        console.print(
            f"  → [bold green]{signal.action}[/bold green] "
            f"price={signal.price:.2f} size={signal.size:.1f} "
            f"confidence={signal.confidence:.0%}"
        )
        console.print(f"    Reasoning: {signal.reasoning[:120]}")

        # Execute trade
        try:
            result = clob.buy(signal.token_id, signal.price, signal.size)
            console.print(f"  → [bold green]Order placed:[/bold green] {result}")
        except Exception as exc:
            log.error("Trade failed for %s: %s", slug, exc)
            console.print(f"  → [bold red]Trade failed:[/bold red] {exc}")


async def run_loop(risk_tolerance: str | None = None) -> None:
    """Run the agent in a continuous loop."""
    risk = risk_tolerance or settings.risk_tolerance
    console.print("[bold]Starting Polymarket Agent[/bold]")
    console.print(f"Risk tolerance: [bold yellow]{risk}[/bold yellow]")
    console.print(f"Scan interval: {settings.scan_interval_seconds}s")
    console.print(f"Max bet: ${settings.max_bet_usdc}")

    # Initialize CLOB client
    try:
        clob.init()
        console.print("[green]CLOB client initialized[/green]")
    except Exception as exc:
        console.print(f"[bold red]CLOB init failed:[/bold red] {exc}")
        console.print("Continuing in read-only mode...")

    strategy = LLMStrategy(risk_tolerance=risk)

    while True:
        try:
            await run_once(strategy)
        except Exception as exc:
            log.error("Agent loop error: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")

        console.print(
            f"\n[dim]Sleeping {settings.scan_interval_seconds}s...[/dim]"
        )
        await asyncio.sleep(settings.scan_interval_seconds)
