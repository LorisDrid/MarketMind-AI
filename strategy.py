"""
strategy.py — Sentiment + Technical backtest strategy runner.

Signal logic per 1-hour bar
────────────────────────────
  BUY  : sentiment_score > 0.3  AND  close_price > SMA_20
  SELL : sentiment_score < -0.1  OR  (close_price < SMA_20 AND position held)

Position sizing
───────────────
  BUY  → BUY_SHARES shares; falls back to CASH_FRACTION of cash if underfunded
  SELL → full position (all shares held)

Look-ahead bias prevention
───────────────────────────
  OHLCV  : rolling SMA window — step i only sees bars 0..i
  News   : each bar receives only headlines published AT OR BEFORE that
           bar's close timestamp.  yfinance publish timestamps are fetched
           directly (data_manager.get_ticker_news strips them for its cache
           format), so we call yf.Ticker.news once here and filter per step.
  Caching: if the news set is unchanged between two bars, the previous
           sentiment score is reused to avoid redundant LLM calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from data_manager import get_historical_data
from engine import Engine
from sentiment_analyzer import get_sentiment

logger = logging.getLogger(__name__)

# ── Strategy constants ────────────────────────────────────────────────────────

SMA_WINDOW    = 20       # bars for the Simple Moving Average
BUY_SHARES    = 5.0      # fixed position size per buy signal
CASH_FRACTION = 0.10     # fallback: fraction of cash to deploy
SENTIMENT_BUY  = 0.3     # score must exceed this to trigger BUY
SENTIMENT_SELL = -0.1    # score must be below this to trigger SELL
STARTING_CASH  = 10_000.0
DB_PATH        = "data/backtest.db"


# ── News helpers (timestamp-aware) ───────────────────────────────────────────


def _fetch_timestamped_news(ticker: str) -> list[dict]:
    """Fetch raw yfinance news preserving publish timestamps.

    data_manager.get_ticker_news normalises headlines for caching but strips
    timestamps.  We call yfinance directly here so the strategy loop can
    filter by publish time, which is necessary to avoid look-ahead bias on
    the news dimension.

    Returns:
        List of dicts:
            ``{"title": str, "publisher": str, "publish_time": datetime | None}``
        ``publish_time`` is UTC-aware when resolvable, else ``None`` (the bar
        loop conservatively treats undated news as "already published").
    """
    try:
        raw_items: list[dict] = yf.Ticker(ticker.upper()).news or []
    except Exception as exc:
        logger.warning("Failed to fetch news for %s: %s", ticker, exc)
        return []

    results: list[dict] = []
    for item in raw_items:
        content = item.get("content", item)      # v2 API wraps under 'content'

        title: str = content.get("title", "").strip()
        if not title:
            continue

        ts: Optional[datetime] = None

        # yfinance v2: content.pubDate / content.displayTime  (ISO-8601 string)
        for key in ("pubDate", "displayTime"):
            raw_str = content.get(key, "")
            if raw_str:
                try:
                    ts = datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
                    break
                except (ValueError, AttributeError):
                    pass

        # yfinance v1: providerPublishTime  (Unix epoch int)
        if ts is None:
            raw_ts = item.get("providerPublishTime") or content.get("providerPublishTime")
            if raw_ts:
                try:
                    ts = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
                except (ValueError, OSError):
                    pass

        publisher: str = (
            content.get("provider", {}).get("displayName", "")
            or content.get("publisher", "")
            or item.get("publisher", "")
        )

        results.append({"title": title, "publisher": publisher, "publish_time": ts})

    logger.info("Fetched %d timestamped news items for %s", len(results), ticker)
    return results


def _headlines_at(news_items: list[dict], cutoff: datetime) -> list[str]:
    """Return headline titles published at or before *cutoff* (UTC).

    News items without a resolvable timestamp are included conservatively —
    they could be old articles, so excluding them would introduce a different
    kind of bias.
    """
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)

    return [
        item["title"]
        for item in news_items
        if item["publish_time"] is None or item["publish_time"] <= cutoff
    ]


# ── Technical indicator ───────────────────────────────────────────────────────


def _compute_sma(series: pd.Series, window: int = SMA_WINDOW) -> pd.Series:
    """Rolling mean with min_periods=1 so early bars still have a value."""
    return series.rolling(window=window, min_periods=1).mean()


# ── Core backtest loop ────────────────────────────────────────────────────────


def run_backtest(
    ticker:        str,
    start_date:    str,
    end_date:      str,
    starting_cash: float = STARTING_CASH,
    period:        str   = "60d",
    interval:      str   = "1h",
    db_path:       str   = DB_PATH,
) -> list[dict]:
    """Simulate trading decisions over historical 1-hour bars.

    Iterates through every bar in ``[start_date, end_date]`` in chronological
    order.  At each step the strategy only "sees" data available up to that
    point in time (no look-ahead bias on either OHLCV or news).

    Args:
        ticker:        Stock symbol, e.g. ``"NVDA"``.
        start_date:    Inclusive simulation start (``"YYYY-MM-DD"``).
        end_date:      Inclusive simulation end   (``"YYYY-MM-DD"``).
        starting_cash: Virtual wallet balance in USD.
        period:        yfinance fetch window — must be wide enough to cover
                       ``start_date`` *and* provide enough bars for SMA_20
                       warm-up (e.g. ``"7d"`` for a 5-day backtest).
        interval:      Bar size — strategy is designed for ``"1h"``.
        db_path:       SQLite file path for trade persistence.

    Returns:
        List of per-bar dicts.  Each dict contains:
        ``timestamp, price, sma_20, sentiment, headlines_used, action,
        shares_held, cash, portfolio_value``.
        Returns an empty list on unrecoverable data failure.
    """
    ticker = ticker.upper()
    logger.info("run_backtest: %s | %s → %s", ticker, start_date, end_date)

    # ── 1. Load OHLCV via data_manager ───────────────────────────────────────
    df = get_historical_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        logger.error("No OHLCV data for %s — aborting backtest.", ticker)
        return []

    # Normalise index to UTC-aware DatetimeIndex
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Slice to the requested simulation window
    ts_start = pd.Timestamp(start_date, tz="UTC")
    ts_end   = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    df = df.loc[(df.index >= ts_start) & (df.index <= ts_end)].copy()

    if df.empty:
        logger.error(
            "OHLCV has no rows in [%s, %s] for %s.", start_date, end_date, ticker
        )
        return []

    # Pre-compute SMA_20 across the sliced window (rolling — no look-ahead)
    df["SMA_20"] = _compute_sma(df["Close"])

    # ── 2. Fetch timestamped news once (filtered per bar) ────────────────────
    news_items = _fetch_timestamped_news(ticker)

    # ── 3. Initialise engine ─────────────────────────────────────────────────
    engine = Engine(starting_cash=starting_cash, db_path=db_path)

    # ── 4. Step through each hourly bar ──────────────────────────────────────
    results: list[dict] = []

    # Sentiment cache: avoids re-querying Ollama when news set is unchanged
    _prev_headlines: frozenset[str] = frozenset()
    _cached_sentiment: float = 0.0

    for timestamp, row in df.iterrows():
        price: float = float(row["Close"])
        sma20: float = float(row["SMA_20"])

        # News available at or before this bar's timestamp (no look-ahead)
        headlines = _headlines_at(news_items, cutoff=timestamp)
        headlines_key = frozenset(headlines)

        # Re-use cached score if the news set hasn't changed
        if headlines_key == _prev_headlines:
            sentiment = _cached_sentiment
        elif headlines:
            sentiment = get_sentiment(headlines, ticker=ticker)
            _prev_headlines  = headlines_key
            _cached_sentiment = sentiment
        else:
            sentiment = 0.0  # neutral when no news available yet

        # ── Current position ─────────────────────────────────────────────────
        portfolio = engine.get_portfolio()
        held_qty  = next(
            (p.quantity for p in portfolio.positions if p.ticker == ticker),
            0.0,
        )
        holding = held_qty > 0
        cash    = portfolio.cash

        # ── Signal evaluation ────────────────────────────────────────────────
        buy_signal  = (sentiment > SENTIMENT_BUY)  and (price > sma20)
        sell_signal = (sentiment < SENTIMENT_SELL) or (holding and price < sma20)

        action       = "HOLD"
        trade_result = None

        if sell_signal and holding:
            trade_result = engine.sell(
                ticker=ticker,
                price=price,
                quantity=held_qty,
                sentiment_score=sentiment,
            )
            action = "SELL" if trade_result.success else "HOLD"

        elif buy_signal and not holding:
            qty = BUY_SHARES
            if price * qty > cash:
                # Fallback: spend at most CASH_FRACTION of available cash
                qty = (cash * CASH_FRACTION) / price if price > 0 else 0.0
            if qty > 0:
                trade_result = engine.buy(
                    ticker=ticker,
                    price=price,
                    quantity=qty,
                    sentiment_score=sentiment,
                )
                action = "BUY" if trade_result.success else "HOLD"

        # ── Snapshot after action ────────────────────────────────────────────
        snap       = engine.get_portfolio()
        held_after = next(
            (p.quantity for p in snap.positions if p.ticker == ticker),
            0.0,
        )
        port_value = snap.cash + held_after * price

        results.append({
            "timestamp":       timestamp.isoformat(),
            "price":           round(price, 4),
            "sma_20":          round(sma20, 4),
            "sentiment":       round(sentiment, 4),
            "headlines_used":  len(headlines),
            "action":          action,
            "shares_held":     round(held_after, 4),
            "cash":            round(snap.cash, 2),
            "portfolio_value": round(port_value, 2),
        })

        logger.debug(
            "[%s] price=%.2f  sma=%.2f  sent=%+.3f  → %-4s  | port=$%.2f",
            str(timestamp)[:16], price, sma20, sentiment, action, port_value,
        )

    return results


# ── Report printer ────────────────────────────────────────────────────────────


def _print_report(ticker: str, results: list[dict], starting_cash: float) -> None:
    """Print a concise backtest summary to stdout."""
    if not results:
        print("No results to display.")
        return

    SEP   = "=" * 64
    first = results[0]
    last  = results[-1]

    total_return = (last["portfolio_value"] - starting_cash) / starting_cash * 100
    n_buy  = sum(1 for r in results if r["action"] == "BUY")
    n_sell = sum(1 for r in results if r["action"] == "SELL")

    print(f"\n{SEP}")
    print(f"  Backtest Report — {ticker}")
    print(SEP)
    print(f"  Period         : {first['timestamp'][:10]}  →  {last['timestamp'][:10]}")
    print(f"  Bars processed : {len(results)}")
    print(f"  Starting cash  : ${starting_cash:>10,.2f}")
    print(f"  Final value    : ${last['portfolio_value']:>10,.2f}")
    print(f"  Total return   : {total_return:>+10.2f}%")
    print(f"  Trades (B / S) : {n_buy} buys  /  {n_sell} sells")
    print(f"  Shares held    : {last['shares_held']}")
    print(f"  Cash remaining : ${last['cash']:>10,.2f}")
    print(SEP)

    print("\n  Last 5 bars:")
    header = f"  {'Timestamp':<20} {'Price':>8} {'SMA20':>8} {'Sent':>6} {'Action':>6} {'PortVal':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results[-5:]:
        print(
            f"  {r['timestamp'][:19]:<20} "
            f"{r['price']:>8.2f} "
            f"{r['sma_20']:>8.2f} "
            f"{r['sentiment']:>+6.3f} "
            f"{r['action']:>6} "
            f"${r['portfolio_value']:>9,.2f}"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import timedelta

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TICKER = "NVDA"
    end    = datetime.now(timezone.utc).date()
    start  = end - timedelta(days=5)

    results = run_backtest(
        ticker=TICKER,
        start_date=str(start),
        end_date=str(end),
        starting_cash=STARTING_CASH,
        period="7d",          # 7-day fetch gives SMA_20 warm-up bars
        db_path=DB_PATH,
    )

    _print_report(TICKER, results, STARTING_CASH)
    sys.exit(0 if results else 1)
