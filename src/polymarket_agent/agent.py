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


async def run_once(
    strategy: LLMStrategy,
    search_terms: list[str] | None = None,
    max_hours: float | None = None,
) -> None:
    """Run one iteration of the agent loop.

    If *search_terms* is given, re-search for the current live market(s)
    matching those terms (so short-term markets auto-rotate to the next
    live window).  Otherwise scan broadly.
    """
    if search_terms is not None:
        console.print(f"\n[bold]Searching live market for: {', '.join(search_terms)}[/bold]")
        markets = await gamma.search_and_filter(terms=search_terms, max_hours=max_hours)
        if not markets:
            console.print("[yellow]No live market found for this window. Waiting for next...[/yellow]")
            return
        console.print(f"Found {len(markets)} live market(s)")
        for m in markets:
            console.print(f"  • {m.get('question', '?')}")
    else:
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


class _SearchConfig:
    """Holds the parsed search parameters so the loop can re-search each iteration."""

    def __init__(self, terms: list[str], max_hours: float | None = None) -> None:
        self.terms = terms
        self.max_hours = max_hours


async def _pick_markets() -> _SearchConfig | None:
    """Ask the user what market to search for in natural language.

    Returns a ``_SearchConfig`` with the parsed terms (so the loop can
    re-search on every iteration and always find the current live market),
    or *None* to scan all markets broadly.
    """
    from polymarket_agent.analyst.llm import parse_query

    console.print(
        "\n[bold]What market do you want to monitor?[/bold]"
        "\n  Examples: [cyan]btc 15min[/cyan], [cyan]eth 1h[/cyan], [cyan]sol 5min[/cyan]"
        "\n  or [bold]all[/bold] to scan everything"
        "\n  or [bold]q[/bold] to quit"
    )

    query = input("\n> ").strip()

    if not query:
        return None
    if query.lower() in ("q", "quit", "exit"):
        return _SearchConfig(terms=[])  # empty terms = abort
    if query.lower() == "all":
        return None

    console.print(f"\nInterpreting: [cyan]{query}[/cyan]")
    parsed = await parse_query(query)
    console.print(f"Search terms: [bold]{', '.join(parsed.search_terms)}[/bold]")

    # Do an initial search to confirm results exist
    console.print("Searching markets...")
    markets = await gamma.search_and_filter(
        terms=parsed.search_terms,
        max_hours=parsed.max_hours,
    )

    if not markets:
        console.print("[yellow]No markets found. Falling back to broad scan.[/yellow]")
        return None

    # Show what was found
    table = Table(title=f"Found {len(markets)} market(s)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Question", style="cyan", max_width=60)
    table.add_column("Volume", justify="right", style="green")
    table.add_column("End Date", style="yellow", max_width=20)

    for i, m in enumerate(markets, 1):
        table.add_row(
            str(i),
            m.get("question", "?")[:60],
            f"${float(m.get('volume', 0) or 0):,.0f}",
            str(m.get("endDate", "?"))[:20],
        )
    console.print(table)

    console.print(
        f"\n[green]Will monitor this search and auto-rotate to the next "
        f"live window each iteration.[/green]"
    )

    return _SearchConfig(terms=parsed.search_terms, max_hours=parsed.max_hours)


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

    # Ask the user which market(s) to focus on
    search_cfg = await _pick_markets()
    if search_cfg is not None and len(search_cfg.terms) == 0:
        console.print("[dim]Aborted.[/dim]")
        return

    strategy = LLMStrategy(risk_tolerance=risk)

    while True:
        try:
            if search_cfg is not None:
                await run_once(strategy, search_terms=search_cfg.terms, max_hours=search_cfg.max_hours)
            else:
                await run_once(strategy)
        except Exception as exc:
            log.error("Agent loop error: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")

        console.print(
            f"\n[dim]Sleeping {settings.scan_interval_seconds}s...[/dim]"
        )
        await asyncio.sleep(settings.scan_interval_seconds)
