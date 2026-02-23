"""
strategy.py — Sentiment + Technical backtest strategy runner.

Signal logic per 1-hour bar
────────────────────────────
  BUY       : sentiment_score >= profile.sentiment_buy
              AND  close_price > SMA_20
              AND  RSI_14 < profile.rsi_overbought  (don't buy into overbought)
  SELL      : sentiment_score <= profile.sentiment_sell
              OR  (close_price < SMA_20 * buffer AND position held)
              OR  (RSI_14 > profile.rsi_overbought AND position held)
              subject to MIN_HOLD_BARS cooldown since the last BUY
  STOP_LOSS : price <= entry_price * (1 − stop_loss_pct)
              fires immediately, bypasses cooldown — highest priority

Position sizing
───────────────
  BUY  → BUY_SHARES shares; falls back to CASH_FRACTION of cash if underfunded
  SELL → full position (all shares held)

Whipsaw / hysteresis protection
─────────────────────────────────
  Buffer zone  : SELL on technical only fires when price drops 0.5 % below SMA_20
                 (not right at the SMA), reducing false reversals on flat markets.
  Cooldown     : MIN_HOLD_BARS bars must elapse after every BUY before any SELL
                 is permitted, preventing rapid round-trips on noisy signals.

Look-ahead bias prevention
───────────────────────────
  OHLCV  : rolling SMA window — step i only sees bars 0..i
  News   : each bar receives only headlines published AT OR BEFORE that
           bar's close timestamp.  yfinance publish timestamps are fetched
           directly (data_manager.get_ticker_news strips them for its cache
           format), so we call yf.Ticker.news once here and filter per step.
  Caching: if the news set is unchanged between two bars, the previous
           sentiment score is reused to avoid redundant LLM calls.

CLI usage
─────────
  python strategy.py --ticker NVDA --days 7
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from database import get_connection
from data_manager import get_historical_data
from engine import Engine, TRANSACTION_FEE
from sentiment_analyzer import get_sentiment

logger = logging.getLogger(__name__)

# ── Strategy constants ────────────────────────────────────────────────────────

SMA_WINDOW     = 20       # bars for the Simple Moving Average
BUY_SHARES     = 5.0      # fixed position size per buy signal
CASH_FRACTION  = 0.10     # fallback: fraction of cash to deploy
SENTIMENT_BUY  = 0.3      # score must exceed this to trigger BUY
SENTIMENT_SELL = -0.1     # score must be below this to trigger SELL
MIN_HOLD_BARS  = 4        # minimum bars held before a SELL is permitted
STARTING_CASH  = 10_000.0
DB_PATH        = "data/backtest.db"

# ── Strategy profiles ─────────────────────────────────────────────────────────
# Each profile is a self-contained parameter set; all three share the same
# signal logic but differ in aggressiveness / risk tolerance.

PROFILES: dict[str, dict] = {
    "Prudent": {
        "sentiment_buy":   0.50,   # high conviction: needs strong positive signal
        "sentiment_sell":  -0.20,  # only exits on clearly negative sentiment
        "sma_window":      20,     # standard 20-bar SMA
        "sma_buffer":      0.990,  # 1 % below SMA before technical SELL fires
        "min_hold_bars":   6,      # longer cooldown — avoids shaking out early
        "buy_shares":      3.0,    # smaller position = lower exposure
        "stop_loss_pct":   0.03,   # 3 % stop-loss — wider buffer to avoid noise
        "rsi_window":      14,     # standard Wilder RSI period
        "rsi_oversold":    35,     # more conservative: wait for deeper oversold
        "rsi_overbought":  65,     # exit sooner
    },
    "Balanced": {
        "sentiment_buy":   0.30,   # current production defaults
        "sentiment_sell":  -0.10,
        "sma_window":      20,
        "sma_buffer":      0.995,  # 0.5 % buffer
        "min_hold_bars":   4,
        "buy_shares":      5.0,
        "stop_loss_pct":   0.02,   # 2 % stop-loss
        "rsi_window":      14,
        "rsi_oversold":    30,     # standard RSI thresholds
        "rsi_overbought":  70,
    },
    "Aggressive": {
        "sentiment_buy":   0.15,   # enters on weak positive signal
        "sentiment_sell":  -0.05,  # exits quickly on any negative drift
        "sma_window":      10,     # faster SMA reacts sooner
        "sma_buffer":      0.998,  # near-zero buffer — tight stops
        "min_hold_bars":   2,      # short cooldown allows rapid re-entry
        "buy_shares":      8.0,    # larger position = higher upside / downside
        "stop_loss_pct":   0.015,  # 1.5 % stop-loss — tightest risk control
        "rsi_window":      14,
        "rsi_oversold":    35,     # more frequent RSI signals
        "rsi_overbought":  65,
    },
}

DEFAULT_PROFILE = "Balanced"

# Tickers that trade continuously (no market-session gaps in OHLCV data).
# yfinance returns 24/7 hourly bars for these; no special SMA treatment is
# needed because the rolling window operates on consecutive bars regardless
# of whether gaps exist — but we log the mode for clarity.
_CRYPTO_TICKERS: frozenset[str] = frozenset({"BTC-USD", "ETH-USD"})


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


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (RSI) over *series*.

    Uses exponential smoothing (``alpha = 1/window``, Wilder's method) for
    both the average gain and average loss.  Early bars (fewer than *window*
    completed periods) are filled with ``50.0`` (neutral) to avoid NaN
    propagation during the warm-up phase.

    Args:
        series: Close price series (``pd.Series``).
        window: Look-back period in bars (default 14 — standard Wilder RSI).

    Returns:
        ``pd.Series`` of RSI values in ``[0, 100]``.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    # Avoid division by zero: replace zero avg_loss with NaN so RS = NaN → RSI = 50
    rs  = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


# ── Sim metadata ──────────────────────────────────────────────────────────────


def _store_sim_meta(
    db_path: str,
    profile_name: str,
    ticker: str,
    sim_start: str,
    sim_end: str,
) -> None:
    """Upsert simulation period metadata into ``backtest_meta``.

    Called once per ``run_backtest`` execution so the dashboard equity curve
    can anchor its X-axis to the simulation start rather than the first trade.

    The UPSERT ensures the row for ``(profile_name, ticker)`` always reflects
    the *most recent* run window.
    """
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO backtest_meta (profile_name, ticker, sim_start, sim_end)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(profile_name, ticker) DO UPDATE SET
                    sim_start   = excluded.sim_start,
                    sim_end     = excluded.sim_end,
                    recorded_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                """,
                (profile_name, ticker, sim_start, sim_end),
            )
            conn.commit()
        logger.debug(
            "Stored sim meta: %s / %s [%s → %s]",
            profile_name, ticker, sim_start, sim_end,
        )
    except Exception as exc:
        logger.warning("Could not store sim meta: %s", exc)


# ── Core backtest loop ────────────────────────────────────────────────────────


def run_backtest(
    ticker:        str,
    start_date:    str,
    end_date:      str,
    starting_cash: float = STARTING_CASH,
    period:        str   = "60d",
    interval:      str   = "1h",
    db_path:       str   = DB_PATH,
    profile_name:  str   = DEFAULT_PROFILE,
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
        profile_name:  One of the keys in ``PROFILES`` (default: "Balanced").

    Returns:
        List of per-bar dicts.  Each dict contains:
        ``timestamp, price, sma_20, sentiment, headlines_used, action,
        shares_held, cash, portfolio_value``.
        Returns an empty list on unrecoverable data failure.
    """
    ticker = ticker.upper()
    _mode = "24/7 crypto" if ticker in _CRYPTO_TICKERS else "equity"
    logger.info(
        "run_backtest: %s | %s → %s  [profile: %s | mode: %s]",
        ticker, start_date, end_date, profile_name, _mode,
    )
    if ticker in _CRYPTO_TICKERS:
        logger.info(
            "%s is a crypto ticker: OHLCV bars are continuous 24/7 — "
            "no market-session gaps; SMA and signals operate on all bars.",
            ticker,
        )

    # ── 0. Resolve profile parameters ────────────────────────────────────────
    profile          = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])
    p_sent_buy       = profile["sentiment_buy"]
    p_sent_sell      = profile["sentiment_sell"]
    p_sma_window     = profile["sma_window"]
    p_sma_buffer     = profile["sma_buffer"]
    p_min_hold       = profile["min_hold_bars"]
    p_buy_shares     = profile["buy_shares"]
    p_stop_loss      = profile["stop_loss_pct"]
    p_rsi_window     = profile["rsi_window"]
    p_rsi_overbought = profile["rsi_overbought"]

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

    # Pre-compute SMA and RSI across the sliced window (rolling — no look-ahead)
    df["SMA_20"] = _compute_sma(df["Close"], window=p_sma_window)
    df["RSI_14"] = _compute_rsi(df["Close"], window=p_rsi_window)

    # ── 2. Fetch timestamped news once (filtered per bar) ────────────────────
    news_items = _fetch_timestamped_news(ticker)

    # ── 3. Initialise engine (tagged with profile so every trade is labelled) ──
    engine = Engine(
        starting_cash=starting_cash,
        db_path=db_path,
        strategy_name=profile_name,
    )

    # ── 3b. Record simulation period metadata ─────────────────────────────────
    # Stored now (before any trades) so the dashboard equity curve can anchor
    # its X-axis to the simulation start even when zero trades were executed.
    _store_sim_meta(db_path, profile_name, ticker, start_date, end_date)

    # ── 4. Step through each hourly bar ──────────────────────────────────────
    results: list[dict] = []

    # Sentiment cache: avoids re-querying Ollama when news set is unchanged
    _prev_headlines: frozenset[str] = frozenset()
    _cached_sentiment: float = 0.0

    # Cooldown counter: bars elapsed since the last successful BUY.
    # A SELL is only permitted once this reaches MIN_HOLD_BARS.
    bars_since_buy: int = 0

    # Stop-loss tracking: entry_price is set on every BUY, cleared on every exit.
    entry_price: float = 0.0

    for timestamp, row in df.iterrows():
        price: float = float(row["Close"])
        sma20: float = float(row["SMA_20"])
        rsi14: float = float(row["RSI_14"])

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

        # ── Signal evaluation (profile-parameterised) ────────────────────────

        # Stop-loss: absolute priority — fires immediately, bypasses cooldown.
        # Only active when we hold a position and have a recorded entry price.
        stop_loss_triggered = (
            holding
            and entry_price > 0.0
            and price <= entry_price * (1.0 - p_stop_loss)
        )

        buy_signal = (
            (sentiment >= p_sent_buy)
            and (price > sma20)
            and (rsi14 < p_rsi_overbought)   # don't buy into an already overbought market
        )
        # Buffer zone: price must fall p_sma_buffer below SMA before technical SELL
        # RSI overbought adds an additional exit condition
        sell_signal = (
            (sentiment <= p_sent_sell)
            or (holding and price < sma20 * p_sma_buffer)
            or (holding and rsi14 > p_rsi_overbought)
        )
        # Cooldown: block sentiment/technical SELL until p_min_hold bars have elapsed
        cooldown_ok = bars_since_buy >= p_min_hold

        action       = "HOLD"
        trade_result = None

        if stop_loss_triggered:
            # Stop-loss fires regardless of cooldown — protect capital first
            _saved_entry = entry_price  # capture before clearing
            trade_result = engine.sell(
                ticker=ticker,
                price=price,
                quantity=held_qty,
                sentiment_score=sentiment,
                timestamp=timestamp.isoformat(),
            )
            if trade_result.success:
                action         = "STOP_LOSS"
                bars_since_buy = 0
                entry_price    = 0.0
                drop_pct       = (1.0 - price / _saved_entry) * 100
                logger.info(
                    "[%s] STOP-LOSS triggered: price=%.2f  entry=%.2f  drop=%.2f%%",
                    str(timestamp)[:16], price, _saved_entry, drop_pct,
                )

        elif sell_signal and holding and cooldown_ok:
            trade_result = engine.sell(
                ticker=ticker,
                price=price,
                quantity=held_qty,
                sentiment_score=sentiment,
                timestamp=timestamp.isoformat(),
            )
            action = "SELL" if trade_result.success else "HOLD"
            if action == "SELL":
                bars_since_buy = 0   # reset counter after position is closed
                entry_price    = 0.0

        elif buy_signal and not holding:
            qty = p_buy_shares
            # Account for transaction fee in the affordability check
            if price * qty + TRANSACTION_FEE > cash:
                # Fallback: spend at most CASH_FRACTION of available cash
                qty = (cash * CASH_FRACTION) / price if price > 0 else 0.0
            if qty > 0:
                trade_result = engine.buy(
                    ticker=ticker,
                    price=price,
                    quantity=qty,
                    sentiment_score=sentiment,
                    timestamp=timestamp.isoformat(),
                )
                action = "BUY" if trade_result.success else "HOLD"
                if action == "BUY":
                    bars_since_buy = 0      # start cooldown clock on open
                    entry_price    = price  # record entry for stop-loss tracking

        # ── Snapshot after action ────────────────────────────────────────────
        snap       = engine.get_portfolio()
        held_after = next(
            (p.quantity for p in snap.positions if p.ticker == ticker),
            0.0,
        )
        port_value = snap.cash + held_after * price

        # Advance the cooldown counter on every bar where a position is held
        if held_after > 0:
            bars_since_buy += 1

        results.append({
            "timestamp":       timestamp.isoformat(),
            "price":           round(price, 4),
            "sma_20":          round(sma20, 4),
            "rsi_14":          round(rsi14, 2),
            "sentiment":       round(sentiment, 4),
            "headlines_used":  len(headlines),
            "action":          action,
            "shares_held":     round(held_after, 4),
            "cash":            round(snap.cash, 2),
            "portfolio_value": round(port_value, 2),
        })

        logger.debug(
            "[%s] price=%.2f  sma=%.2f  rsi=%.1f  sent=%+.3f  → %-9s  | port=$%.2f",
            str(timestamp)[:16], price, sma20, rsi14, sentiment, action, port_value,
        )

    return results


# ── Report printer ────────────────────────────────────────────────────────────


def _print_report(
    ticker: str,
    results: list[dict],
    starting_cash: float,
    profile_name: str = "",
) -> None:
    """Print a concise backtest summary to stdout."""
    if not results:
        print("No results to display.")
        return

    SEP   = "=" * 64
    first = results[0]
    last  = results[-1]

    total_return = (last["portfolio_value"] - starting_cash) / starting_cash * 100
    n_buy       = sum(1 for r in results if r["action"] == "BUY")
    n_sell      = sum(1 for r in results if r["action"] == "SELL")
    n_stop_loss = sum(1 for r in results if r["action"] == "STOP_LOSS")

    print(f"\n{SEP}")
    label = f"  [{profile_name}]" if profile_name else ""
    print(f"  Backtest Report — {ticker}{label}")
    print(SEP)
    print(f"  Period         : {first['timestamp'][:10]}  →  {last['timestamp'][:10]}")
    print(f"  Bars processed : {len(results)}")
    print(f"  Starting cash  : ${starting_cash:>10,.2f}")
    print(f"  Final value    : ${last['portfolio_value']:>10,.2f}")
    print(f"  Total return   : {total_return:>+10.2f}%")
    print(f"  Trades (B / S) : {n_buy} buys  /  {n_sell} sells  /  {n_stop_loss} stop-losses")
    print(f"  Shares held    : {last['shares_held']}")
    print(f"  Cash remaining : ${last['cash']:>10,.2f}")
    print(SEP)

    print("\n  Last 5 bars:")
    header = f"  {'Timestamp':<20} {'Price':>8} {'SMA20':>8} {'RSI':>5} {'Sent':>6} {'Action':>10} {'PortVal':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results[-5:]:
        print(
            f"  {r['timestamp'][:19]:<20} "
            f"{r['price']:>8.2f} "
            f"{r['sma_20']:>8.2f} "
            f"{r.get('rsi_14', 0):>5.1f} "
            f"{r['sentiment']:>+6.3f} "
            f"{r['action']:>10} "
            f"${r['portfolio_value']:>9,.2f}"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    from datetime import timedelta
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MarketMind AI — Backtest runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker", default="NVDA",
        help="Stock ticker symbol to simulate",
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of calendar days to simulate (end = today)",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=list(PROFILES.keys()),
        help="Strategy profile to run",
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help="SQLite output path (shared across profiles)",
    )
    args = parser.parse_args()

    # ── Per-profile DB reset ──────────────────────────────────────────────────
    # Removes only this profile's previous trades so other profiles' data is
    # preserved for comparison.  portfolio/positions are always reset so the
    # engine starts with a clean $10 k slate regardless of prior runs.
    bt_path = Path(args.db)
    if bt_path.exists():
        with get_connection(bt_path) as _conn:
            _conn.execute(
                "DELETE FROM trades WHERE strategy_name = ?", (args.profile,)
            )
            _conn.execute("DELETE FROM portfolio")
            _conn.execute("DELETE FROM positions")
            _conn.execute(
                "INSERT INTO portfolio (current_cash) VALUES (?)", (STARTING_CASH,)
            )
            # Clear stale sim-period metadata for this profile so the equity
            # curve X-axis reflects the new run, not a previous one.
            try:
                _conn.execute(
                    "DELETE FROM backtest_meta WHERE profile_name = ?", (args.profile,)
                )
            except Exception:
                pass  # table may not exist on a very first run
            _conn.commit()
        logger.info("Reset profile '%s' in %s", args.profile, bt_path)

    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=args.days)
    # Extra fetch window ensures SMA warm-up bars exist before the sim window
    fetch_period = f"{args.days + 5}d"

    results = run_backtest(
        ticker=args.ticker,
        start_date=str(start),
        end_date=str(end),
        starting_cash=STARTING_CASH,
        period=fetch_period,
        db_path=args.db,
        profile_name=args.profile,
    )

    _print_report(args.ticker, results, STARTING_CASH, profile_name=args.profile)
    sys.exit(0 if results else 1)
