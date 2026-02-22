# MarketMind AI — Paper Trading Simulator

An AI-powered paper trading simulator that combines **technical analysis** (SMA crossover) with **LLM-based sentiment scoring** via Claude to make simulated trading decisions on historical data.

> **Status:** MVP scaffolding — database layer and broker engine are complete. Data fetching, strategy, and UI are next.

---

## Architecture

```
marketmind/
├── database.py       # SQLite schema & connection factory
├── engine.py         # Virtual broker (buy/sell, PnL, positions)
├── data_manager.py   # [TODO] yfinance data fetching & caching
├── strategy.py       # [TODO] SMA crossover + Claude sentiment
├── app.py            # [TODO] Streamlit dashboard
├── data/             # SQLite DB + cached CSVs (gitignored)
├── logs/             # Application logs (gitignored)
├── .env.example      # API key template
└── requirements.txt
```

### Database schema

| Table       | Purpose                                          |
|-------------|--------------------------------------------------|
| `portfolio` | Append-only cash ledger — latest row is balance  |
| `positions` | One row per open position (updated in-place)     |
| `trades`    | Immutable transaction log with sentiment scores  |

---

## Setup

### 1. Clone & create the virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

---

## What you can test right now

### Database initialisation

```bash
python database.py
```

Expected output:
```
INFO: Initialising database at data/marketmind.db
INFO: Seeded portfolio with starting cash: $10000.00
INFO: Database ready.
```

Re-running is idempotent — it will not re-seed the portfolio.

### Broker engine (buy / sell / portfolio)

```bash
python engine.py
```

Expected output:
```
INFO: Initial portfolio: cash=$10000.00, positions=0
INFO: BUY 10.0000 AAPL @ $190.5000 (total $1905.00) → cash $8095.00
INFO: SELL 5.0000 AAPL @ $195.0000 (proceeds $975.00) → cash $9070.00
INFO: Final portfolio: cash=$9070.00, positions=[('AAPL', 5.0)]
```

This validates the full write path: schema creation → buy → weighted avg cost basis → partial sell → cash reconciliation.

### Verify the SQLite database directly

```bash
# With sqlite3 CLI
sqlite3 data/marketmind.db "SELECT * FROM trades;"
sqlite3 data/marketmind.db "SELECT * FROM portfolio ORDER BY id DESC LIMIT 5;"
sqlite3 data/marketmind.db "SELECT * FROM positions;"
```

---

## What's not testable yet

| Module            | Blocker                                        |
|-------------------|------------------------------------------------|
| `data_manager.py` | Not created — no live/cached prices available  |
| `strategy.py`     | Needs `data_manager` + `ANTHROPIC_API_KEY`     |
| `app.py`          | Needs all modules above                        |

### Next steps (in order)

1. **`data_manager.py`** — fetch OHLCV data via `yfinance`, cache to CSV, expose `get_latest_price(ticker)` and `get_news_headlines(ticker)`
2. **`strategy.py`** — SMA 20/50 crossover signal + `get_sentiment_score(ticker)` calling Claude (`claude-opus-4-6`)
3. **`app.py`** — Streamlit dashboard: portfolio curve vs S&P 500, open positions table, trade log

---

## Tech stack

| Layer      | Library                          |
|------------|----------------------------------|
| Data       | `yfinance`, `pandas`             |
| Indicators | `pandas-ta`                      |
| Database   | `sqlite3` (stdlib)               |
| Validation | `pydantic`                       |
| AI         | `anthropic` (`claude-opus-4-6`)  |
| UI         | `streamlit`, `plotly`            |
