"""Main autonomous agent loop with multi-asset price accumulation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from polymarket_agent.analyst.llm import (
    clear_all_window_state,
    get_order_history,
    get_reasoning_history,
    record_order,
    record_reasoning,
)
from polymarket_agent.crypto_prices import (
    PriceTracker,
    extract_asset_from_slug,
    get_spot_price,
    parse_event_start_time,
)
from polymarket_agent.clob import client as clob
from polymarket_agent.config import settings
from polymarket_agent.data import client as data
from polymarket_agent.gamma import client as gamma
from polymarket_agent.strategies.llm_strategy import LLMStrategy

log = logging.getLogger(__name__)
console = Console()

# Minimum % of window that must have elapsed before analysis (by risk level)
_MIN_ELAPSED_PCT: dict[str, float] = {
    "conservative": 0.60,
    "moderate": 0.40,
    "aggressive": 0.15,
}


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


def _market_key(market: dict) -> str:
    """Canonical key for a market (for dedup / order tracking)."""
    return market.get("conditionId") or market.get("condition_id") or market.get("question", "")


def _parse_window_times(market: dict) -> tuple[datetime | None, datetime | None]:
    """Return (start_dt, end_dt) for a market window."""
    slug = market.get("slug", "")
    start_ts = parse_event_start_time(market)
    end_date = market.get("endDate", "")
    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc) if start_ts else None
    end_dt = None
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    return start_dt, end_dt


async def _analyze_asset(
    strategy: LLMStrategy,
    market: dict,
    positions: list[dict],
    price_tracker: PriceTracker,
    asset: str,
) -> dict | None:
    """Run LLM analysis for a single asset/market. Returns signal dict or None.

    Always records reasoning from the analysis so the AI builds context
    across cycles, even when the trade is blocked by low confidence.
    """
    mk = _market_key(market)
    tracker_data = price_tracker.get_history(asset)
    reasoning = get_reasoning_history(mk)

    try:
        signal, analysis = await strategy.evaluate(
            market,
            positions,
            price_tracker_data=tracker_data,
            reasoning_chain=reasoning,
        )
    except Exception as exc:
        log.error("Error evaluating %s: %s", asset, exc)
        return None

    # Always record AI reasoning so it accumulates across cycles
    last_spot = tracker_data[-1]["spot_price"] if tracker_data else None
    last_ref = tracker_data[-1].get("reference_price") if tracker_data else None
    record_reasoning(
        market_key=mk,
        recommendation=analysis.recommendation,
        confidence=analysis.confidence,
        reasoning=analysis.reasoning,
        spot_price=last_spot,
        reference_price=last_ref,
    )

    if signal is None:
        if analysis.recommendation == "SKIP":
            console.print(f"  [{asset.upper()}] → SKIP ({analysis.confidence:.0%} conf) — {analysis.reasoning[:80]}")
        else:
            console.print(
                f"  [{asset.upper()}] → [bold yellow]{analysis.recommendation} BLOCKED[/bold yellow] "
                f"(conf {analysis.confidence:.0%} < min {settings.min_confidence:.0%}) "
                f"— {analysis.reasoning[:80]}"
            )
        return None

    return {"asset": asset, "market": market, "signal": signal, "market_key": mk}


async def _price_collection_task(
    asset_markets: dict[str, dict],
    traded_assets: set[str],
    reference_prices: dict[str, float | None],
    price_tracker: PriceTracker,
    stop_event: asyncio.Event,
) -> None:
    """Continuously collect price samples every price_sample_interval seconds."""
    from polymarket_agent.crypto_prices import get_price_at_time

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        active_assets = set(asset_markets.keys()) - traded_assets

        for asset in active_assets:
            spot = await get_spot_price(asset)
            if spot is not None:
                # Try to resolve reference price if still missing
                if reference_prices.get(asset) is None:
                    start_dt, _ = _parse_window_times(asset_markets[asset])
                    if start_dt and now >= start_dt:
                        # Window has started — retry Binance first
                        start_ts = parse_event_start_time(asset_markets[asset])
                        ref_price = await get_price_at_time(asset, start_ts) if start_ts else None
                        if ref_price is not None:
                            reference_prices[asset] = ref_price
                            console.print(
                                f"  [{asset.upper()}] Reference price (Binance) set to ${ref_price:,.2f}"
                            )
                        else:
                            # Binance still unavailable, use current spot as fallback
                            reference_prices[asset] = spot
                            console.print(
                                f"  [{asset.upper()}] Reference price (fallback) set to ${spot:,.2f}"
                            )
                    # If before window start, leave as None — keep showing N/A
                ref = reference_prices.get(asset)
                snap = price_tracker.record_snapshot(asset, spot, ref)
                diff_str = f"${snap['diff']:+,.2f}" if snap.get("diff") is not None else "N/A"
                console.print(
                    f"  [dim][{snap['timestamp']}][/dim] {asset.upper()} "
                    f"${spot:,.2f} (diff: {diff_str})"
                )

        # Wait for next sample (or until stopped)
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=settings.price_sample_interval
            )
        except asyncio.TimeoutError:
            pass  # Continue to next iteration


async def _analysis_task(
    strategy: LLMStrategy,
    asset_markets: dict[str, dict],
    traded_assets: set[str],
    price_tracker: PriceTracker,
    positions: list[dict],
    min_pct: float,
    stop_event: asyncio.Event,
) -> None:
    """Run LLM analysis every analysis_interval seconds for assets that are ready."""
    cycle = 0
    first_run = True

    while not stop_event.is_set():
        if not first_run:
            # Wait for the analysis interval (skip wait on first run)
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=settings.analysis_interval
                )
                if stop_event.is_set():
                    break
            except asyncio.TimeoutError:
                pass  # Continue to analysis
        first_run = False

        now = datetime.now(timezone.utc)
        active_assets = set(asset_markets.keys()) - traded_assets
        if not active_assets:
            continue

        cycle += 1

        # Check elapsed % for each asset
        ready_for_analysis: list[str] = []
        for asset in active_assets:
            m = asset_markets[asset]
            start_dt, end_dt = _parse_window_times(m)
            if start_dt and end_dt:
                window_secs = (end_dt - start_dt).total_seconds()
                elapsed_secs = (now - start_dt).total_seconds()
                elapsed_pct = elapsed_secs / window_secs if window_secs > 0 else 0
                if elapsed_pct >= min_pct:
                    ready_for_analysis.append(asset)
                else:
                    mins_elapsed = elapsed_secs / 60
                    mins_total = window_secs / 60
                    console.print(
                        f"  [{asset.upper()}] {mins_elapsed:.0f}m/{mins_total:.0f}m "
                        f"({elapsed_pct:.0%} < {min_pct:.0%}) — accumulating"
                    )
            else:
                # No timing info, allow analysis
                ready_for_analysis.append(asset)

        if not ready_for_analysis:
            continue

        console.print(
            f"\n[bold blue]Analysis cycle {cycle} — evaluating: "
            f"{', '.join(a.upper() for a in ready_for_analysis)}[/bold blue]"
        )

        # Run analysis in parallel for all ready assets
        tasks = []
        for asset in ready_for_analysis:
            if asset in traded_assets:
                continue
            m = asset_markets[asset]
            # Check duplicate guard
            mk = _market_key(m)
            if get_order_history(mk):
                console.print(f"  [{asset.upper()}] Already traded this window. Skipping.")
                traded_assets.add(asset)
                continue
            tasks.append(_analyze_asset(strategy, m, positions, price_tracker, asset))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    log.error("Analysis task failed: %s", result)
                    continue
                if result is None:
                    continue
                if not isinstance(result, dict):
                    continue

                asset = result["asset"]
                signal = result["signal"]
                market = result["market"]
                mk = result["market_key"]

                console.print(
                    f"  [{asset.upper()}] → [bold green]{signal.action}[/bold green] "
                    f"price={signal.price:.2f} size={signal.size:.1f} "
                    f"confidence={signal.confidence:.0%}"
                )
                console.print(f"    Reasoning: {signal.reasoning[:120]}")

                # Execute trade
                max_price = round(signal.price + 0.03, 2)
                try:
                    order_result = clob.market_buy(signal.token_id, signal.size, max_price=max_price)
                    console.print(f"  [{asset.upper()}] [bold green]Order placed:[/bold green] {order_result}")
                    record_order(
                        market_key=mk,
                        action=signal.action,
                        token_id=signal.token_id,
                        size=signal.size,
                        price=signal.price,
                    )
                    traded_assets.add(asset)
                except Exception as exc:
                    log.error("Trade failed for %s: %s", asset, exc)
                    console.print(f"  [{asset.upper()}] [bold red]Trade failed:[/bold red] {exc}")


async def _monitor_window(
    strategy: LLMStrategy,
    markets: list[dict],
    price_tracker: PriceTracker,
) -> datetime | None:
    """Run concurrent price collection and analysis loops for one set of markets.

    Price collection runs every price_sample_interval seconds (continuous).
    Analysis runs every analysis_interval seconds (after min_elapsed_pct is met).
    Both run in parallel so prices continue to be collected even during LLM processing.

    Exits when all assets are traded or the earliest window expires.
    Returns the earliest window end time so the caller can wait for it.
    """
    if not markets:
        return None

    risk = strategy.risk_tolerance or settings.risk_tolerance
    min_pct = _MIN_ELAPSED_PCT.get(risk, 0.40)

    # Build asset → market mapping
    asset_markets: dict[str, dict] = {}
    for m in markets:
        asset = extract_asset_from_slug(m.get("slug", ""))
        if asset:
            asset_markets[asset] = m

    if not asset_markets:
        console.print("[yellow]No crypto up/down markets found in this set.[/yellow]")
        return None

    traded_assets: set[str] = set()  # assets that have been traded this window

    # Determine earliest window end
    earliest_end: datetime | None = None
    for m in asset_markets.values():
        _, end_dt = _parse_window_times(m)
        if end_dt:
            if earliest_end is None or end_dt < earliest_end:
                earliest_end = end_dt

    # Get reference prices (once at start)
    from polymarket_agent.crypto_prices import get_price_at_time

    reference_prices: dict[str, float | None] = {}
    for asset, m in asset_markets.items():
        start_ts = parse_event_start_time(m)
        if start_ts:
            reference_prices[asset] = await get_price_at_time(asset, start_ts)
        else:
            reference_prices[asset] = None

    positions: list[dict] = []
    try:
        positions = await data.get_positions()
    except Exception as exc:
        log.warning("Could not fetch positions: %s", exc)

    console.print(f"\n[bold]Monitoring {len(asset_markets)} asset(s):[/bold] {', '.join(a.upper() for a in asset_markets)}")
    console.print(f"Risk: {risk} | Min elapsed: {min_pct:.0%} | Earliest end: {earliest_end}")
    console.print(f"Price sampling: every {settings.price_sample_interval}s | Analysis: every {settings.analysis_interval}s")

    # Create stop event for coordinating task shutdown
    stop_event = asyncio.Event()

    # Launch concurrent tasks
    price_task = asyncio.create_task(
        _price_collection_task(asset_markets, traded_assets, reference_prices, price_tracker, stop_event)
    )
    analysis_task = asyncio.create_task(
        _analysis_task(strategy, asset_markets, traded_assets, price_tracker, positions, min_pct, stop_event)
    )

    try:
        # Monitor exit conditions
        while True:
            await asyncio.sleep(1)  # Check every second

            now = datetime.now(timezone.utc)

            # Check if window expired
            if earliest_end and now >= earliest_end:
                console.print("\n[bold yellow]Window expired. Moving to next window.[/bold yellow]")
                stop_event.set()
                break

            # Check if all assets traded
            active_assets = set(asset_markets.keys()) - traded_assets
            if not active_assets:
                console.print("\n[bold green]All assets traded this window.[/bold green]")
                stop_event.set()
                break

    finally:
        # Ensure tasks are stopped and cleaned up
        stop_event.set()
        await asyncio.gather(price_task, analysis_task, return_exceptions=True)

    return earliest_end


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
        "\n  Examples: [cyan]btc 15min[/cyan], [cyan]eth 1h[/cyan], [cyan]15min[/cyan] (all crypto)"
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
    table.add_column("Asset", style="bold", width=6)
    table.add_column("Volume", justify="right", style="green")
    table.add_column("End Date", style="yellow", max_width=20)

    for i, m in enumerate(markets, 1):
        asset = extract_asset_from_slug(m.get("slug", "")) or "?"
        table.add_row(
            str(i),
            m.get("question", "?")[:60],
            asset.upper(),
            f"${float(m.get('volume', 0) or 0):,.0f}",
            str(m.get("endDate", "?"))[:20],
        )
    console.print(table)

    console.print(
        f"\n[green]Will monitor {len(markets)} market(s) and auto-rotate to the next "
        f"live window each iteration.[/green]"
    )

    return _SearchConfig(terms=parsed.search_terms, max_hours=parsed.max_hours)


async def run_loop(risk_tolerance: str | None = None) -> None:
    """Run the agent in a continuous loop with price accumulation."""
    risk = risk_tolerance or settings.risk_tolerance
    console.print("[bold]Starting Polymarket Agent[/bold]")
    console.print(f"Risk tolerance: [bold yellow]{risk}[/bold yellow]")
    console.print(f"Price sampling: every {settings.price_sample_interval}s | Analysis: every {settings.analysis_interval}s")
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
    price_tracker = PriceTracker()

    while True:
        try:
            # Search for current live markets
            if search_cfg is not None:
                console.print(f"\n[bold]Searching live markets for: {', '.join(search_cfg.terms)}[/bold]")
                markets = await gamma.search_and_filter(
                    terms=search_cfg.terms, max_hours=search_cfg.max_hours
                )
                if not markets:
                    console.print("[yellow]No live market found. Waiting 30s for next window...[/yellow]")
                    await asyncio.sleep(30)
                    continue
                console.print(f"Found {len(markets)} live market(s)")
                for m in markets:
                    asset = extract_asset_from_slug(m.get("slug", "")) or "?"
                    console.print(f"  • [{asset.upper()}] {m.get('question', '?')}")
            else:
                console.print("\n[bold]Scanning markets...[/bold]")
                raw_markets = await scan_markets(limit=50)
                markets = raw_markets
                console.print(f"Found {len(markets)} markets passing filters")

            # Clear per-window state and price history for fresh start
            clear_all_window_state()
            price_tracker.clear_all()

            # Run the two-phase monitor for this window
            window_end = await _monitor_window(strategy, markets, price_tracker)

            # Wait until the window actually expires before searching for the next one
            if window_end:
                now = datetime.now(timezone.utc)
                wait_secs = (window_end - now).total_seconds()
                if wait_secs > 0:
                    console.print(
                        f"\n[dim]Window ends in {wait_secs:.0f}s. "
                        f"Waiting for expiry before next search...[/dim]"
                    )
                    await asyncio.sleep(wait_secs + 5)  # +5s buffer
                    continue

        except Exception as exc:
            log.error("Agent loop error: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")

        # Brief pause before searching for next window
        console.print(f"\n[dim]Searching for next window in 10s...[/dim]")
        await asyncio.sleep(10)
