"""Main autonomous agent loop."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from polymarket_agent.analyst.llm import get_order_history, record_order
from polymarket_agent.crypto_prices import extract_asset_from_slug, parse_event_start_time
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

    # Minimum % of window that must have elapsed before trading (by risk level)
    _MIN_ELAPSED_PCT: dict[str, float] = {
        "conservative": 0.60,  # wait until 60% of window elapsed
        "moderate": 0.40,      # wait until 40% of window elapsed
        "aggressive": 0.15,    # wait until 15% of window elapsed
    }

    for market in markets:
        question = market.get("question", "?")
        slug = market.get("slug", "")
        console.print(f"\n[bold blue]Evaluating:[/bold blue] {question}")

        # Time guard for short-term markets: don't trade too early in the window
        if extract_asset_from_slug(slug):
            start_ts = parse_event_start_time(market)
            end_date = market.get("endDate", "")
            if start_ts and end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)
                    now = datetime.now(timezone.utc)
                    window_secs = (end_dt - start_dt).total_seconds()
                    elapsed_secs = (now - start_dt).total_seconds()

                    if window_secs > 0 and elapsed_secs >= 0:
                        elapsed_pct = elapsed_secs / window_secs
                        risk = strategy.risk_tolerance or settings.risk_tolerance
                        min_pct = _MIN_ELAPSED_PCT.get(risk, 0.40)

                        if elapsed_pct < min_pct:
                            mins_elapsed = elapsed_secs / 60
                            mins_total = window_secs / 60
                            console.print(
                                f"  → [bold yellow]TOO EARLY:[/bold yellow] "
                                f"{mins_elapsed:.0f}m/{mins_total:.0f}m elapsed "
                                f"({elapsed_pct:.0%} < {min_pct:.0%} required for {risk} risk). Waiting."
                            )
                            continue
                except (ValueError, TypeError):
                    pass

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
            f"price={signal.price:.2f} max={round(signal.price + 0.03, 2):.2f} "
            f"size={signal.size:.1f} confidence={signal.confidence:.0%}"
        )
        console.print(f"    Reasoning: {signal.reasoning[:120]}")

        # Hard guard: skip if we already have an order on this market this session
        market_key = market.get("conditionId") or market.get("condition_id") or market.get("question", "")
        past_orders = get_order_history(market_key)
        if past_orders:
            console.print(
                f"  → [bold yellow]BLOCKED:[/bold yellow] Already placed "
                f"{len(past_orders)} order(s) on this market this session. Skipping."
            )
            continue

        # Execute trade (market order — FOK with price ceiling)
        # Allow up to 3 cents of slippage above the LLM's suggested price
        max_price = round(signal.price + 0.03, 2)
        try:
            result = clob.market_buy(signal.token_id, signal.size, max_price=max_price)
            console.print(f"  → [bold green]Order placed:[/bold green] {result}")
            record_order(
                market_key=market_key,
                action=signal.action,
                token_id=signal.token_id,
                size=signal.size,
                price=signal.price,
            )
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
