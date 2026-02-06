# Polymarket Agent

Autonomous trading agent for [Polymarket](https://polymarket.com). Discovers prediction markets, analyzes them with LLMs, and executes trades automatically.

## Quickstart

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Installation

```bash
# 1. Clone or download the repository
git clone <repository-url>
cd polymarket-agent

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the package and dependencies
pip install -e .

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys and wallet details
```

### Configuration

Edit `.env` and add at minimum:

```env
# For scanning and basic commands (no trading):
# (No configuration needed)

# For LLM analysis (analyze, ask commands):
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://opencode.ai/zen/v1
LLM_MODEL=opencode/claude-sonnet-4-5

# For trading (run command):
PRIVATE_KEY=0x...
WALLET_ADDRESS=0x...
```

### Basic Usage

```bash
# Scan markets (no wallet or LLM required)
python -m polymarket_agent scan

# Analyze a specific market by slug (requires LLM)
python -m polymarket_agent analyze will-gta-6-cost-100

# Ask in natural language (requires LLM)
python -m polymarket_agent ask "analyze btc 15min"
python -m polymarket_agent ask "show me crypto markets"
python -m polymarket_agent ask "what political bets are there?"

# View your positions (requires wallet address)
python -m polymarket_agent portfolio

# Run the autonomous trading agent (requires everything)
python -m polymarket_agent run
```

## Natural language queries

The `ask` command lets you search and analyze markets using plain language. The LLM interprets your request, searches Polymarket, and optionally analyzes the results.

```bash
# Search for specific market types
python -m polymarket_agent ask "show me btc 15min markets"
python -m polymarket_agent ask "eth up or down 1h"
python -m polymarket_agent ask "what solana bets are active?"

# Search + analyze
python -m polymarket_agent ask "analyze btc 15min"
python -m polymarket_agent ask "analyze crypto markets aggressively"
python -m polymarket_agent ask "analyze trump election markets conservatively"

# Time-filtered search (markets expiring soon)
python -m polymarket_agent ask "show me markets expiring in 24h"
```

The agent understands Polymarket's short-term "Up or Down" markets (BTC, ETH, SOL, XRP in 5min/15min/1h intervals) and maps keywords like "btc 15min" to the correct market type.

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Wallet (EOA / MetaMask)
PRIVATE_KEY=0x...
WALLET_ADDRESS=0x...

# LLM — compatible with OpenCode Zen, OpenRouter, OpenAI direct, Anthropic direct
LLM_PROVIDER=openai              # "openai" or "anthropic"
LLM_API_KEY=sk-...
LLM_BASE_URL=https://opencode.ai/zen/v1
LLM_MODEL=opencode/claude-sonnet-4-5

# Risk tolerance (natural language — the LLM adapts its analysis to match)
RISK_TOLERANCE=moderate

# Hard trading limits (enforced in code, LLM cannot exceed these)
MAX_BET_USDC=50          # Max per trade
MAX_PORTFOLIO_USDC=500   # Max total exposure
MIN_CONFIDENCE=0.3       # Absolute min confidence floor
SCAN_INTERVAL_SECONDS=300
MIN_LIQUIDITY=1000
MIN_VOLUME=5000
```

### Supported LLM providers

| Provider | `LLM_PROVIDER` | `LLM_BASE_URL` | `LLM_MODEL` (example) |
|----------|----------------|-----------------|------------------------|
| OpenCode Zen | `openai` | `https://opencode.ai/zen/v1` | `opencode/claude-sonnet-4-5` |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4-5-20250929` |
| OpenAI direct | `openai` | `https://api.openai.com/v1` | `gpt-4o` |
| Anthropic direct | `anthropic` | *(leave empty)* | `claude-sonnet-4-5-20250929` |

## Architecture

```
src/polymarket_agent/
├── config.py              # Settings from .env (pydantic-settings)
├── gamma/client.py        # Gamma API → market discovery + smart search
├── clob/client.py         # CLOB API → trade execution
├── data/client.py         # Data API → positions and portfolio
├── analyst/
│   ├── llm.py             # Multi-provider LLM client + query interpreter
│   └── prompts.py         # Prompts for analysis + query parsing
├── strategies/
│   ├── base.py            # Strategy protocol + TradeSignal
│   └── llm_strategy.py    # LLM strategy with risk filters
├── agent.py               # Autonomous loop
└── cli.py                 # CLI
```

### Agent flow

```
┌─────────────┐     ┌──────────┐     ┌─────────────────┐     ┌──────────┐
│  Gamma API  │────▶│ Filters  │────▶│   LLM analysis  │────▶│   CLOB   │
│  (markets)  │     │ vol/liq  │     │ (risk-adapted)  │     │  (trade) │
└─────────────┘     └──────────┘     └─────────────────┘     └──────────┘
                                            │
                                       ┌────┴────┐
                                       │Data API │
                                       │(posit.) │
                                       └─────────┘
```

### `ask` flow

```
User query → LLM query interpreter → search_terms + filters
  → Gamma API (tag-based search) → text filter → active filter
    → Display results
    → If action=analyze → LLM analysis per market
```

## Risk management

Risk is **LLM-adaptive**: you describe your tolerance in natural language and the LLM adjusts its analysis to match.

### Setting your risk tolerance

| Method | Example |
|--------|---------|
| `.env` default | `RISK_TOLERANCE=moderate` |
| `--risk` flag (overrides .env) | `--risk "aggressive, I want high returns"` |
| Inside `ask` queries | `ask "analyze btc aggressively"` |

```bash
# Conservative — only trade on very clear mispricings
python -m polymarket_agent run --risk "conservative, only bet when edge is >15 cents"

# Aggressive — trade more often, accept smaller edges
python -m polymarket_agent run --risk "aggressive, maximize returns, I accept drawdowns"

# Domain-specific
python -m polymarket_agent run --risk "moderate on politics, aggressive on sports"
```

### What the LLM adapts

- **Minimum edge**: conservative = needs large mispricings, aggressive = acts on small edges
- **Position sizing**: conservative = smaller sizes, aggressive = larger
- **Confidence threshold**: conservative = only recommends when very confident, aggressive = acts on weaker signals
- **Reasoning**: always explains how risk profile influenced the decision

### Hard safety caps

These limits are enforced in code regardless of what the LLM suggests:

| Limit | Setting | Default |
|-------|---------|---------|
| Max per trade | `MAX_BET_USDC` | $50 |
| Max total exposure | `MAX_PORTFOLIO_USDC` | $500 |
| Min confidence floor | `MIN_CONFIDENCE` | 0.3 |

## Commands

| Command | Description | Requires |
|---------|-------------|----------|
| `scan` | List active markets passing filters | Nothing |
| `analyze <slug> [--risk ...]` | LLM analysis of a specific market | LLM config |
| `ask <query>` | Natural language search + analysis | LLM config |
| `portfolio` | Show current positions | `WALLET_ADDRESS` |
| `run [--risk ...]` | Full autonomous loop | Everything |

## Requirements

- Python 3.11+
- EOA wallet with USDC on Polygon for trading
- API key for an LLM provider

## Disclaimer

This software is experimental. Trading on prediction markets carries risk of loss. Use at your own risk.
