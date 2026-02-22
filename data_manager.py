"""
data_manager.py - Market data provider.

Fetches OHLCV price history and news headlines via yfinance.
Results are cached locally in data/cache/ to avoid redundant API calls.

Cache TTL
---------
- Historical OHLCV : 1 hour  (HISTORICAL_CACHE_TTL)
- News headlines   : 1 hour  (NEWS_CACHE_TTL)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path(__file__).parent / "data" / "cache"
HISTORICAL_CACHE_TTL: int = 3_600   # seconds
NEWS_CACHE_TTL: int = 3_600         # seconds


# ---------------------------------------------------------------------------
# Internal cache helpers
# ---------------------------------------------------------------------------


def _hist_cache_path(ticker: str, period: str, interval: str) -> Path:
    return CACHE_DIR / f"{ticker}_{period}_{interval}.csv"


def _news_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}_news.json"


def _cache_is_fresh(path: Path, ttl: int) -> bool:
    """Return True if *path* exists and its mtime is within *ttl* seconds."""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_historical_data(
    ticker: str,
    period: str = "60d",
    interval: str = "1h",
) -> Optional[pd.DataFrame]:
    """Return a cleaned OHLCV DataFrame for *ticker*.

    Columns returned: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
    The index is a timezone-aware ``DatetimeIndex``.

    Data is cached as a CSV in ``data/cache/``. A fresh cache file (younger
    than :data:`HISTORICAL_CACHE_TTL` seconds) is returned immediately
    without hitting the network.

    Args:
        ticker:   Stock symbol, e.g. ``"NVDA"``.
        period:   yfinance period string, e.g. ``"60d"``, ``"1y"``.
        interval: yfinance interval string, e.g. ``"1h"``, ``"1d"``.

    Returns:
        Cleaned :class:`~pandas.DataFrame`, or ``None`` on failure.
    """
    ticker = ticker.upper()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _hist_cache_path(ticker, period, interval)

    # --- Cache hit ---
    if _cache_is_fresh(cache_path, HISTORICAL_CACHE_TTL):
        logger.info("Cache hit  : %s historical (%s / %s)", ticker, period, interval)
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.debug("Loaded %d rows from cache → %s", len(df), cache_path.name)
            return df
        except Exception as exc:
            logger.warning(
                "Cache read failed for %s (%s). Re-fetching. Error: %s",
                ticker, cache_path.name, exc,
            )

    # --- Fetch from yfinance ---
    logger.info("Fetching   : %s historical (%s / %s) via yfinance", ticker, period, interval)
    try:
        raw: pd.DataFrame = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=True,   # Adjusts for splits/dividends automatically
        )
    except Exception as exc:
        logger.error("yfinance request failed for '%s': %s", ticker, exc, exc_info=True)
        return None

    if raw.empty:
        logger.error(
            "yfinance returned no data for '%s'. "
            "Ticker may be invalid or delisted.",
            ticker,
        )
        return None

    # Keep only standard OHLCV columns (yfinance sometimes adds extras)
    ohlcv = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    df = raw[ohlcv].copy()
    df.dropna(how="all", inplace=True)

    # Persist to cache
    try:
        df.to_csv(cache_path)
        logger.info("Cached     : %d rows for %s → %s", len(df), ticker, cache_path.name)
    except Exception as exc:
        logger.warning("Could not write cache for %s: %s", ticker, exc)

    return df


def get_latest_price(ticker: str) -> Optional[float]:
    """Return the most recent market price for *ticker*.

    Uses :pyattr:`yfinance.FastInfo.last_price` for a lightweight,
    near-real-time quote (no caching needed — the call is a single HTTP
    request). Falls back to the tail of the cached/fetched OHLCV history
    if ``FastInfo`` is unavailable.

    Args:
        ticker: Stock symbol, e.g. ``"NVDA"``.

    Returns:
        Price as a ``float``, or ``None`` on failure.
    """
    ticker = ticker.upper()

    # --- Primary: FastInfo (fast, live) ---
    try:
        price: Optional[float] = yf.Ticker(ticker).fast_info.last_price
        if price is not None and price > 0:
            logger.info("Latest price %s: $%.4f (fast_info)", ticker, price)
            return float(price)
        logger.warning("fast_info.last_price returned %r for %s", price, ticker)
    except Exception as exc:
        logger.warning(
            "fast_info failed for '%s' (%s). Falling back to history.", ticker, exc
        )

    # --- Fallback: tail of OHLCV history ---
    df = get_historical_data(ticker, period="5d", interval="1h")
    if df is not None and not df.empty:
        price = float(df["Close"].dropna().iloc[-1])
        logger.info("Latest price %s: $%.4f (history fallback)", ticker, price)
        return price

    logger.error("Could not determine latest price for '%s'.", ticker)
    return None


def get_ticker_news(ticker: str, max_items: int = 10) -> list[dict]:
    """Return recent news headlines for *ticker*.

    Each item is a dict with three keys:

    * ``"title"``     — headline text
    * ``"link"``      — canonical article URL
    * ``"publisher"`` — outlet display name

    Results are cached as JSON in ``data/cache/`` for
    :data:`NEWS_CACHE_TTL` seconds.

    Args:
        ticker:    Stock symbol, e.g. ``"NVDA"``.
        max_items: Maximum number of headlines to return.

    Returns:
        List of headline dicts; empty list on failure.
    """
    ticker = ticker.upper()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _news_cache_path(ticker)

    # --- Cache hit ---
    if _cache_is_fresh(cache_path, NEWS_CACHE_TTL):
        logger.info("Cache hit  : %s news", ticker)
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning(
                "News cache read failed for %s: %s. Re-fetching.", ticker, exc
            )

    # --- Fetch from yfinance ---
    logger.info("Fetching   : %s news via yfinance", ticker)
    try:
        raw_news: list[dict] = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.error("yfinance news request failed for '%s': %s", ticker, exc, exc_info=True)
        return []

    # yfinance 1.x wraps each item under a 'content' key:
    #   item = { "id": "...", "content": { "title": ..., "provider": {...}, ... } }
    headlines: list[dict] = []
    for item in raw_news[:max_items]:
        content: dict = item.get("content", item)   # graceful fallback for older shapes

        title: str = content.get("title", "").strip()
        if not title:
            continue

        # Prefer canonicalUrl → clickThroughUrl
        link: str = (
            content.get("canonicalUrl", {}).get("url", "")
            or content.get("clickThroughUrl", {}).get("url", "")
            or content.get("link", "")
            or item.get("link", "")
        )

        publisher: str = (
            content.get("provider", {}).get("displayName", "")
            or content.get("publisher", "")
            or item.get("publisher", "")
        )

        headlines.append({"title": title, "link": link, "publisher": publisher})

    if not headlines:
        logger.warning("No news headlines found for '%s'.", ticker)

    # Persist to cache
    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(headlines, fh, ensure_ascii=False, indent=2)
        logger.info("Cached     : %d headlines for %s → %s", len(headlines), ticker, cache_path.name)
    except Exception as exc:
        logger.warning("Could not write news cache for %s: %s", ticker, exc)

    return headlines


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TICKER = "NVDA"
    SEP = "=" * 56

    print(f"\n{SEP}")
    print(f"  MarketMind — Data Manager Smoke Test  ({TICKER})")
    print(f"{SEP}\n")

    # 1. Latest price
    price = get_latest_price(TICKER)
    print(f"Latest price : ${price:,.4f}\n" if price else f"Latest price : FAILED\n")

    # 2. Historical data (tail)
    df = get_historical_data(TICKER, period="5d", interval="1h")
    if df is not None:
        print(f"OHLCV data   : {len(df)} rows  |  last 3:")
        print(df.tail(3).to_string())
        print()
    else:
        print("OHLCV data   : FAILED\n")

    # 3. News (top 3)
    news = get_ticker_news(TICKER)
    print(f"News ({len(news)} total, showing 3):")
    for i, item in enumerate(news[:3], 1):
        print(f"  {i}. [{item['publisher']}] {item['title']}")
        print(f"     {item['link']}")
    print()
