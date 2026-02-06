# Polymarket Autonomous Trading Agent

## What is this

Autonomous trading agent for Polymarket. Discovers prediction markets, analyzes them with LLMs, and executes trades automatically.

## Stack

- Python 3.11+, installable package with `pip install -e .`
- `pydantic-settings` for config from `.env`
- `httpx` (async) for Gamma API and Data API
- `py-clob-client` for trading (CLOB API, EOA wallet)
- `openai` / `anthropic` SDKs for LLM analysis (compatible with OpenCode Zen, OpenRouter, direct)
- `rich` for CLI output

## Project structure

```
src/polymarket_agent/
├── config.py              # Settings (pydantic-settings, .env)
├── gamma/client.py        # Gamma API: market discovery + smart search
├── clob/client.py         # CLOB API: trade execution (py-clob-client)
├── data/client.py         # Data API: positions and portfolio
├── analyst/
│   ├── llm.py             # Multi-provider LLM client + query interpreter
│   └── prompts.py         # Prompt templates for analysis + query parsing
├── strategies/
│   ├── base.py            # Strategy protocol + TradeSignal model
│   └── llm_strategy.py    # LLM-based strategy with risk filters
├── agent.py               # Main autonomous loop
├── cli.py                 # CLI entry point
└── __main__.py            # python -m polymarket_agent
```

## CLI commands

```bash
python -m polymarket_agent scan                                    # Scan markets (no wallet required)
python -m polymarket_agent analyze <slug>                          # Analyze market with LLM
python -m polymarket_agent analyze <slug> --risk "conservative"    # Analyze with custom risk tolerance
python -m polymarket_agent portfolio                               # View current positions
python -m polymarket_agent run                                     # Full autonomous loop
python -m polymarket_agent run --risk "aggressive, high returns"   # Run with custom risk tolerance
python -m polymarket_agent ask "analyze btc 15min"                 # Natural language search + analysis
python -m polymarket_agent ask "show me crypto markets"            # Natural language search only
```

## Natural language queries (`ask` command)

The `ask` command accepts natural language and uses the LLM to interpret what the user wants, then searches and optionally analyzes markets.

### How it works

1. User input → LLM query interpreter (`parse_query()` in `analyst/llm.py`)
2. LLM extracts: `search_terms`, `max_hours`, `action` (search/analyze), `risk_tolerance`
3. `search_and_filter()` in `gamma/client.py` uses tag-based discovery + text filtering
4. If action is "analyze" → runs LLM analysis on each found market

### Query interpreter (`analyst/llm.py:parse_query`)

- Input: natural language string
- Output: `ParsedQuery` with `search_terms`, `max_hours`, `action`, `risk_tolerance`
- Uses `QUERY_SYSTEM_PROMPT` in `analyst/prompts.py`
- Knows about Polymarket-specific market types (e.g. "BTC Up or Down 15min" markets)

### Smart search (`gamma/client.py:search_and_filter`)

- Maps keywords to Polymarket `tag_slug` values for efficient API queries
- Tag mapping in `TAG_SLUGS` dict: e.g. `"btc"` → `["bitcoin", "crypto-prices"]`, `"15min"` → `["15M", "up-or-down"]`
- Text-filters results against keywords in question, event title, and slug
- Filters out expired markets via `is_still_active()`
- Optionally filters by `max_hours` (expiration time window)
- Deduplicates by market ID, sorts by volume

### Polymarket short-term markets

Polymarket has "Up or Down" markets for crypto (BTC, ETH, SOL, XRP) in 5min, 15min, and 1h intervals:
- Slug pattern: `btc-updown-15m-{timestamp}`, `eth-updown-5m-{timestamp}`, etc.
- Tag slugs: `15M`, `5M`, `1H`, `up-or-down`, `bitcoin`, `crypto-prices`
- These are created on a rolling basis and expire quickly
- When user says "btc 15min" they mean these markets, NOT a time filter

## Configuration (.env)

Required variables by command:

| Variable | Required for | Description |
|----------|-------------|-------------|
| `PRIVATE_KEY` | `run` | EOA private key (0x...) |
| `WALLET_ADDRESS` | `run`, `portfolio` | Wallet address |
| `LLM_PROVIDER` | `analyze`, `ask`, `run` | `"openai"` or `"anthropic"` |
| `LLM_API_KEY` | `analyze`, `ask`, `run` | Provider API key |
| `LLM_BASE_URL` | `analyze`, `ask`, `run` | Base URL (Zen, OpenRouter, etc.) |
| `LLM_MODEL` | `analyze`, `ask`, `run` | Model to use |

Optional variables with defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `RISK_TOLERANCE` | `moderate` | Natural language risk profile injected into LLM system prompt |
| `MAX_BET_USDC` | 50 | Hard cap per trade (enforced in code, LLM cannot exceed) |
| `MAX_PORTFOLIO_USDC` | 500 | Hard cap total exposure (enforced in code) |
| `MIN_CONFIDENCE` | 0.3 | Absolute min confidence floor (enforced in code) |
| `SCAN_INTERVAL_SECONDS` | 300 | Interval between scans |
| `MIN_LIQUIDITY` | 1000 | Min market liquidity |
| `MIN_VOLUME` | 5000 | Min market volume |

## External APIs

- **Gamma API** (`https://gamma-api.polymarket.com`): Market discovery. No auth required. Endpoints: `/events` (with `tag_slug` for categories), `/markets`
- **CLOB API** (`https://clob.polymarket.com`): Trading. Requires EOA wallet with signature. Handles GTC and FOK orders
- **Data API** (`https://data-api.polymarket.com`): Read positions and trades. Endpoints: `/positions`, `/trades`

### Gamma API search notes

- The `_q` parameter on `/markets` is unreliable for keyword search
- Best approach: search `/events` by `tag_slug` → extract markets → text-filter by keyword
- Known tag slugs: `crypto`, `crypto-prices`, `bitcoin`, `up-or-down`, `15M`, `5M`, `1H`, `politics`, `us-elections`, `sports`, `ai`, `science-tech`, `pop-culture`
- Events contain nested `markets` array with full market data
- `active=True&closed=False` doesn't filter out expired short-term markets — use `is_still_active()` to check `endDate`

## Risk management

Risk is **LLM-adaptive**: the user describes their tolerance in natural language and the LLM adjusts its analysis accordingly.

### How it works

1. The user provides a risk tolerance string (e.g. `"conservative"`, `"aggressive, maximize returns"`, `"only sports, moderate risk"`)
2. This string is injected into the LLM **system prompt** via `build_system_prompt()` in `analyst/prompts.py`
3. The LLM adapts: minimum edge required, suggested position size, confidence thresholds, and explains the influence in its reasoning
4. **Hard caps** in code (`MAX_BET_USDC`, `MAX_PORTFOLIO_USDC`, `MIN_CONFIDENCE`) are always enforced regardless of LLM output — the LLM cannot bypass them

### Three ways to set risk tolerance

| Method | Precedence | Example |
|--------|-----------|---------|
| `--risk` CLI flag | Highest | `--risk "aggressive, I accept large drawdowns"` |
| Natural language in `ask` | Via LLM extraction | `ask "analyze btc aggressively"` |
| `RISK_TOLERANCE` in `.env` | Default | `RISK_TOLERANCE=conservative` |
| Hardcoded fallback | Lowest | `"moderate"` (in `config.py`) |

### Data flow

```
risk_tolerance string
  → build_system_prompt() in analyst/prompts.py
    → injected into LLM system prompt
      → LLM returns MarketAnalysis (with adapted confidence, size, reasoning)
        → LLMStrategy enforces hard caps (max bet, max portfolio, min confidence)
          → TradeSignal or None
```

### Key files

- `config.py`: `risk_tolerance` setting (default `"moderate"`)
- `analyst/prompts.py`: `build_system_prompt(risk_tolerance)` — builds system prompt with risk block
- `analyst/prompts.py`: `QUERY_SYSTEM_PROMPT` — system prompt for natural language query interpretation
- `analyst/llm.py`: `analyze_market(..., risk_tolerance=)` — passes risk to prompt builder
- `analyst/llm.py`: `parse_query(user_input)` — interprets natural language into `ParsedQuery`
- `strategies/llm_strategy.py`: `LLMStrategy(risk_tolerance=)` — holds risk for all evaluations
- `agent.py`: `run_loop(risk_tolerance=)` — receives from CLI, passes to strategy
- `cli.py`: `--risk` flag parsing on `run` and `analyze`, query handling in `ask`
- `gamma/client.py`: `TAG_SLUGS` mapping, `search_and_filter()`, `is_still_active()`

## Agent flow (`run`)

1. Scans active markets via Gamma API
2. Filters by minimum volume and liquidity
3. For each market: gets price (CLOB) + positions (Data API)
4. Evaluates with LLM strategy (risk-adapted prompt) → `TradeSignal` or `None`
5. Enforces hard caps on size and confidence
6. If signal passes all filters → executes order via CLOB
7. Waits `SCAN_INTERVAL_SECONDS` and repeats

## Code conventions

- API functions are `async` (gamma, data, analyst). CLOB client is synchronous (wraps py-clob-client)
- Data models use `pydantic.BaseModel` (`MarketAnalysis`, `TradeSignal`, `ParsedQuery`)
- `Strategy` is a `Protocol` with method `async evaluate(market, positions) -> TradeSignal | None`
- Config singleton: `from polymarket_agent.config import settings`
- LLM response parsing expects raw JSON (or with markdown fences)

## How to extend

- **New strategy**: Create class in `strategies/` implementing the `Strategy` protocol from `strategies/base.py`
- **New LLM provider**: Add `_call_<provider>` function in `analyst/llm.py` and case in `chat()`
- **New CLI command**: Add case in `cli.py:main()` and `_<command>()` function
- **New market category**: Add entry to `TAG_SLUGS` in `gamma/client.py`
- **Market filters**: Modify `gamma/client.py:passes_filters()` or settings
