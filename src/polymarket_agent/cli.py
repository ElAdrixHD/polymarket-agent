"""CLI entry point."""

from __future__ import annotations

import asyncio
import logging
import sys

from rich.console import Console

console = Console()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) < 2:
        _usage()
        return

    command = sys.argv[1]

    if command == "run":
        _run()
    elif command == "scan":
        _scan()
    elif command == "portfolio":
        _portfolio()
    elif command == "analyze":
        if len(sys.argv) < 3 or sys.argv[2].startswith("--"):
            console.print("[red]Usage: polymarket-agent analyze <market-slug>[/red]")
            sys.exit(1)
        _analyze(sys.argv[2])
    elif command == "ask":
        query = " ".join(sys.argv[2:])
        if not query.strip():
            console.print("[red]Usage: polymarket-agent ask <natural language query>[/red]")
            sys.exit(1)
        _ask(query)
    else:
        _usage()


def _usage() -> None:
    console.print("[bold]Polymarket Agent[/bold]\n")
    console.print("Commands:")
    console.print("  run                                 Run the autonomous agent loop")
    console.print("  scan                                Scan markets without trading")
    console.print("  portfolio                           Show current positions")
    console.print("  analyze <slug>                      Analyze a specific market")
    console.print("  ask <query>                         Natural language market search & analysis")
    console.print()
    console.print("Ask examples:")
    console.print('  ask analyze btc bets expiring in 15min')
    console.print('  ask show me crypto markets closing in 1h')
    console.print('  ask what political bets are there?')


def _run() -> None:
    from polymarket_agent.agent import run_loop

    asyncio.run(run_loop())


def _scan() -> None:
    from polymarket_agent.agent import print_markets, scan_markets

    async def _do() -> None:
        markets = await scan_markets(limit=50)
        if not markets:
            console.print("[yellow]No markets found matching filters.[/yellow]")
            return
        print_markets(markets)

    asyncio.run(_do())


def _portfolio() -> None:
    from rich.table import Table

    from polymarket_agent.data import client as data

    async def _do() -> None:
        positions = await data.get_positions()
        if not positions:
            console.print("[yellow]No open positions.[/yellow]")
            return

        table = Table(title="Portfolio Positions")
        table.add_column("Market", style="cyan", max_width=50)
        table.add_column("Outcome", style="bold")
        table.add_column("Size", justify="right", style="green")
        table.add_column("Avg Price", justify="right")
        table.add_column("Current", justify="right", style="yellow")

        total = 0.0
        for pos in positions:
            size = float(pos.get("size", 0) or 0)
            cur_price = float(pos.get("currentPrice", 0) or 0)
            total += size * cur_price
            table.add_row(
                str(pos.get("title", pos.get("market", "?")))[:50],
                pos.get("outcome", "?"),
                f"{size:.1f}",
                f"${float(pos.get('avgPrice', 0) or 0):.2f}",
                f"${cur_price:.2f}",
            )

        console.print(table)
        console.print(f"\n[bold]Total value:[/bold] ${total:,.2f}")

    asyncio.run(_do())


def _analyze(slug: str) -> None:
    from polymarket_agent.analyst.llm import analyze_market
    from polymarket_agent.gamma import client as gamma

    async def _do() -> None:
        console.print(f"Fetching market: [cyan]{slug}[/cyan]")
        market = await gamma.get_market(slug)
        console.print(f"Question: [bold]{market.get('question', '?')}[/bold]")

        console.print("Analyzing with LLM...")
        analysis = await analyze_market(market)

        console.print(f"\n[bold]Recommendation:[/bold] {analysis.recommendation}")
        console.print(f"[bold]Confidence:[/bold] {analysis.confidence:.0%}")
        console.print(f"[bold]Estimated Probability:[/bold] {analysis.estimated_probability:.0%}")
        console.print(f"[bold]Reasoning:[/bold] {analysis.reasoning}")
        if analysis.suggested_price is not None:
            console.print(f"[bold]Suggested Price:[/bold] ${analysis.suggested_price:.2f}")
        if analysis.suggested_size is not None:
            console.print(f"[bold]Suggested Size:[/bold] {analysis.suggested_size:.1f}")

    asyncio.run(_do())


def _ask(query: str) -> None:
    from rich.table import Table

    from polymarket_agent.analyst.llm import analyze_market, parse_query
    from polymarket_agent.gamma import client as gamma

    async def _do() -> None:
        console.print(f"Interpreting: [cyan]{query}[/cyan]")
        parsed = await parse_query(query)

        console.print(f"Search terms: [bold]{', '.join(parsed.search_terms)}[/bold]")
        if parsed.max_hours is not None:
            if parsed.max_hours < 1:
                console.print(f"Time filter: expiring within [yellow]{parsed.max_hours * 60:.0f} minutes[/yellow]")
            else:
                console.print(f"Time filter: expiring within [yellow]{parsed.max_hours:.0f} hours[/yellow]")
        console.print(f"Action: [bold]{parsed.action}[/bold]")
        console.print("\nSearching markets...")
        markets = await gamma.search_and_filter(
            terms=parsed.search_terms,
            max_hours=parsed.max_hours,
        )

        if not markets:
            console.print("[yellow]No markets found matching your query.[/yellow]")
            if parsed.max_hours is not None:
                console.print("[dim]Try removing the time constraint or broadening your search.[/dim]")
            return

        # Display found markets
        table = Table(title=f"Found {len(markets)} market(s)")
        table.add_column("#", style="dim", width=3)
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

        # If action is analyze, run LLM analysis on each market
        if parsed.action == "analyze":
            console.print(f"\n[bold]Analyzing {len(markets)} market(s) with LLM...[/bold]")
            for i, m in enumerate(markets, 1):
                question = m.get("question", "?")
                console.print(f"\n[bold blue]({i}/{len(markets)}) {question}[/bold blue]")
                try:
                    analysis = await analyze_market(m)
                    rec_color = "green" if analysis.recommendation != "SKIP" else "dim"
                    console.print(f"  Recommendation: [{rec_color}]{analysis.recommendation}[/{rec_color}]")
                    console.print(f"  Confidence: {analysis.confidence:.0%}")
                    console.print(f"  Est. probability: {analysis.estimated_probability:.0%}")
                    console.print(f"  Reasoning: {analysis.reasoning[:150]}")
                    if analysis.suggested_price is not None:
                        console.print(f"  Suggested price: ${analysis.suggested_price:.2f}")
                    if analysis.suggested_size is not None:
                        console.print(f"  Suggested size: {analysis.suggested_size:.1f}")
                except Exception as exc:
                    console.print(f"  [red]Analysis failed: {exc}[/red]")

    asyncio.run(_do())
