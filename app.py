"""
app.py — MarketMind AI · Streamlit Trading Dashboard  v2

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_manager import get_historical_data, get_latest_price, get_ticker_news
from database import get_connection
from engine import Engine

# ── Page config (must be the very first Streamlit call) ───────────────────
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MarketMind AI — Paper Trading Simulator v2"},
)

logging.basicConfig(level=logging.WARNING)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }

        /* Metric cards */
        [data-testid="metric-container"] {
            background   : rgba(255,255,255,0.03);
            border       : 1px solid rgba(255,255,255,0.09);
            border-radius: 0.6rem;
            padding      : 0.85rem 1.1rem;
        }

        /* News publisher chip */
        .pub-chip {
            display      : inline-block;
            font-size    : 0.68rem;
            font-weight  : 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color        : #888;
            background   : rgba(255,255,255,0.06);
            border-radius: 0.3rem;
            padding      : 0.1rem 0.45rem;
            margin-bottom: 0.5rem;
        }

        /* Divider spacing */
        hr { margin: 1.2rem 0 !important; }

        /* Tighter subheader top margin */
        h2, h3 { margin-top: 0.3rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────
TICKERS: list[str] = ["NVDA", "AAPL", "MSFT", "TSLA"]

_PERIOD_INTERVAL: dict[str, str] = {
    "1d" : "5m",
    "5d" : "30m",
    "30d": "1h",
    "60d": "1h",
}

# US market session in Eastern time (matches yfinance America/New_York tz)
_MARKET_OPEN  = 9.5   # 09:30
_MARKET_CLOSE = 16.0  # 16:00

_RANGEBREAKS = [
    dict(bounds=["sat", "mon"]),
    dict(bounds=[_MARKET_CLOSE, _MARKET_OPEN], pattern="hour"),
]

_GREEN = "#2ecc71"
_RED   = "#e74c3c"
_BLUE  = "#4e8df5"
_AMBER = "#f39c12"

BACKTEST_DB_PATH = Path("data/backtest.db")


# ── Utilities ──────────────────────────────────────────────────────────────

def fmt_usd(value: float) -> str:
    """Format a float as a dollar amount: $12,345.67"""
    return f"${value:,.2f}"


def _pnl_delta(pnl: float, pnl_pct: float) -> str:
    """Return a signed delta string that Streamlit can colour correctly.

    Streamlit colours the delta green when it starts with '+' or a digit,
    and red when it starts with '-'.
    """
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{fmt_usd(pnl)} ({pnl_pct:+.2f}%)"


# ── Cached singletons / fetchers ───────────────────────────────────────────

@st.cache_resource
def _engine() -> Engine:
    return Engine()


@st.cache_data          # no TTL — starting cash is fixed for the session
def _initial_cash() -> float:
    """First cash entry in the portfolio table = starting capital."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT current_cash FROM portfolio ORDER BY id ASC LIMIT 1"
        ).fetchone()
    return float(row["current_cash"]) if row else 10_000.0


@st.cache_data(ttl=300, show_spinner="Fetching price history…")
def _history(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    return get_historical_data(ticker, period=period, interval=interval)


@st.cache_data(ttl=60, show_spinner=False)
def _price(ticker: str) -> Optional[float]:
    return get_latest_price(ticker)


@st.cache_data(ttl=600, show_spinner="Fetching news…")
def _news(ticker: str) -> list[dict]:
    return get_ticker_news(ticker)


# ── Chart builder ──────────────────────────────────────────────────────────

def _build_chart(df: pd.DataFrame, ticker: str, chart_type: str) -> go.Figure:
    """Return a Plotly figure with price (+ SMA 20) and volume subplots.

    * Rangebreaks hide weekends and US non-trading hours automatically.
    * Volume bars are green/red depending on candle direction.
    * SMA 20 is overlaid as a dotted amber line.
    """
    cdf = df.copy()
    cdf["SMA20"] = cdf["Close"].rolling(20).mean()

    vol_colors = [
        f"rgba(46,204,113,0.55)" if c >= o else f"rgba(231,76,60,0.55)"
        for c, o in zip(cdf["Close"], cdf["Open"])
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # ── Price ──────────────────────────────────────────────────────────────
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=cdf.index,
                open=cdf["Open"], high=cdf["High"],
                low=cdf["Low"],   close=cdf["Close"],
                name=ticker,
                increasing_line_color=_GREEN, increasing_fillcolor=_GREEN,
                decreasing_line_color=_RED,   decreasing_fillcolor=_RED,
                whiskerwidth=0.5,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=cdf.index, y=cdf["Close"],
                mode="lines", name=ticker,
                line=dict(color=_BLUE, width=1.8),
                fill="tozeroy", fillcolor="rgba(78,141,245,0.07)",
            ),
            row=1, col=1,
        )

    # ── SMA 20 ─────────────────────────────────────────────────────────────
    if not cdf["SMA20"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=cdf.index, y=cdf["SMA20"],
                mode="lines", name="SMA 20",
                line=dict(color=_AMBER, width=1.3, dash="dot"),
            ),
            row=1, col=1,
        )

    # ── Volume ─────────────────────────────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=cdf.index, y=cdf["Volume"],
            name="Volume", marker_color=vol_colors,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#444", font_size=12),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font_size=12,
        ),
    )

    # Rangebreaks applied to all x-axes (both subplots share ET data)
    fig.update_xaxes(
        rangebreaks=_RANGEBREAKS,
        showgrid=False,
        linecolor="rgba(255,255,255,0.1)",
        color="#777",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        color="#888",
        row=1, col=1,
    )
    fig.update_yaxes(
        tickformat=".3s",       # 1.5M, 500K, etc.
        showgrid=False,
        color="#666",
        row=2, col=1,
    )

    return fig


# ── Sentiment gauge ────────────────────────────────────────────────────────

def _sentiment_gauge(score: float = 0.0) -> go.Figure:
    """Compact Plotly gauge for a sentiment score in [-1, 1].

    A score of 0 indicates 'pending' / neutral (Ollama not yet integrated).
    """
    if score > 0.25:
        bar_color = _GREEN
    elif score < -0.25:
        bar_color = _RED
    else:
        bar_color = "#f1c40f"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=22, color="white"), valueformat=".2f"),
        title=dict(text="Market Sentiment", font=dict(size=12, color="#999")),
        gauge=dict(
            axis=dict(
                range=[-1, 1],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["−1", "−0.5", "0", "0.5", "1"],
                tickwidth=1, tickcolor="#555",
                tickfont=dict(size=10, color="#666"),
            ),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[-1,    -0.25], color="rgba(231,76,60,0.12)"),
                dict(range=[-0.25,  0.25], color="rgba(241,196,15,0.08)"),
                dict(range=[ 0.25,  1   ], color="rgba(46,204,113,0.12)"),
            ],
        ),
    ))
    fig.update_layout(
        height=170,
        margin=dict(l=15, r=15, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


# ── Backtest helpers ───────────────────────────────────────────────────────

def _load_backtest_equity(db_path: Path) -> tuple[pd.DataFrame, float]:
    """Reconstruct equity curve from backtest.db by replaying trades.

    Each BUY/SELL in the trades table has a matching cash-snapshot row in the
    portfolio table (inserted atomically by the engine).  Zipping them lets us
    compute ``portfolio_value = cash_after_trade + shares_held * trade_price``
    at every event without storing the value explicitly in the schema.

    Returns:
        (equity_df, init_cash)
        equity_df columns: timestamp (datetime), portfolio_value (float),
                           action ("START" | "BUY" | "SELL")
    """
    with get_connection(db_path) as conn:
        seed = conn.execute(
            "SELECT current_cash FROM portfolio ORDER BY id ASC LIMIT 1"
        ).fetchone()
        init_cash = float(seed["current_cash"]) if seed else 10_000.0

        trades = [
            dict(r) for r in conn.execute(
                "SELECT timestamp, ticker, type, price, quantity "
                "FROM trades ORDER BY id ASC"
            ).fetchall()
        ]
        # Portfolio rows after the seed (one inserted per trade)
        cash_values = [
            float(r["current_cash"])
            for r in conn.execute(
                "SELECT current_cash FROM portfolio ORDER BY id ASC LIMIT -1 OFFSET 1"
            ).fetchall()
        ]

    if not trades:
        return pd.DataFrame(), init_cash

    records = [{"timestamp": pd.NaT, "portfolio_value": init_cash, "action": "START"}]
    shares_held: dict[str, float] = {}

    for trade, cash in zip(trades, cash_values):
        t_ticker = trade["ticker"]
        t_type   = trade["type"]
        price    = float(trade["price"])
        qty      = float(trade["quantity"])

        if t_type == "BUY":
            shares_held[t_ticker] = shares_held.get(t_ticker, 0.0) + qty
        else:
            shares_held[t_ticker] = max(0.0, shares_held.get(t_ticker, 0.0) - qty)

        # For single-ticker backtests (current design): value = cash + held * price
        port_val = cash + shares_held.get(t_ticker, 0.0) * price
        records.append({
            "timestamp":       trade["timestamp"],
            "portfolio_value": round(port_val, 2),
            "action":          t_type,
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df, init_cash


def _build_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    """Plotly line chart of portfolio value with BUY/SELL trade markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"],
        y=equity_df["portfolio_value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(78,141,245,0.07)",
    ))

    buys = equity_df[equity_df["action"] == "BUY"]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["timestamp"], y=buys["portfolio_value"],
            mode="markers", name="BUY",
            marker=dict(color=_GREEN, size=10, symbol="triangle-up"),
        ))

    sells = equity_df[equity_df["action"] == "SELL"]
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["timestamp"], y=sells["portfolio_value"],
            mode="markers", name="SELL",
            marker=dict(color=_RED, size=10, symbol="triangle-down"),
        ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#444", font_size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font_size=12,
        ),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.1)", color="#777")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#888")
    return fig


# ── Session-state init ─────────────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.now()


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 MarketMind AI")
    st.caption("Paper Trading Simulator · v2")
    st.divider()

    # ── Mode selector ──────────────────────────────────────────────────────
    mode: str = st.radio(
        "Mode",
        ["📊 Live", "🔬 Backtest"],
        horizontal=True,
    )
    st.divider()

    # ── Controls ───────────────────────────────────────────────────────────
    ticker: str  = st.selectbox("Ticker", TICKERS, index=0)
    period: str  = st.selectbox("Period", list(_PERIOD_INTERVAL.keys()), index=1)
    interval: str = _PERIOD_INTERVAL[period]
    st.caption(f"Auto interval: `{interval}`")

    chart_type: str = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

    if st.button("🔄  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now()
        st.rerun()

    last_ref: datetime = st.session_state["last_refresh"]
    st.caption(f"⏱ Last updated: **{last_ref.strftime('%H:%M:%S')}**")

    st.divider()

    # ── Open Positions ─────────────────────────────────────────────────────
    st.subheader("Open Positions")
    engine  = _engine()
    portfolio = engine.get_portfolio()

    if portfolio.positions:
        for pos in portfolio.positions:
            live = _price(pos.ticker)
            if live:
                live_val = live * pos.quantity
                cost_val = pos.average_buy_price * pos.quantity
                pnl      = live_val - cost_val
                pnl_pct  = (pnl / cost_val * 100) if cost_val else 0.0
                st.metric(
                    label       =f"{pos.ticker}  ·  {pos.quantity:g} sh",
                    value       =fmt_usd(live_val),
                    delta       =_pnl_delta(pnl, pnl_pct),
                    delta_color ="normal",   # green +, red −
                )
            else:
                st.metric(
                    label=f"{pos.ticker}  ·  {pos.quantity:g} sh",
                    value=fmt_usd(pos.market_value),
                    help ="Live price unavailable — showing cost-basis value.",
                )
    else:
        st.caption("No open positions.")

    st.divider()

    # ── Sentiment Gauge (Ollama placeholder) ───────────────────────────────
    st.subheader("Market Sentiment")
    # score will be injected by strategy.py / Ollama in the next milestone
    sentiment_score: float = st.session_state.get("sentiment_score", 0.0)
    st.plotly_chart(_sentiment_gauge(sentiment_score), use_container_width=True)
    st.caption("⏳ Ollama (local LLM) scoring — coming in `strategy.py`")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════

# ── Backtest mode: render analysis then stop ────────────────────────────────
if mode == "🔬 Backtest":
    st.header("Backtest Analysis")

    if not BACKTEST_DB_PATH.exists():
        st.warning(
            "No backtest data found. "
            "Run `python strategy.py` first to generate `data/backtest.db`."
        )
    else:
        try:
            equity_df, bt_init_cash = _load_backtest_equity(BACKTEST_DB_PATH)

            if equity_df.empty:
                st.info("Backtest database exists but contains no trades yet.")
            else:
                final_val        = equity_df["portfolio_value"].iloc[-1]
                total_return_pct = (final_val - bt_init_cash) / bt_init_cash * 100
                n_trades         = int((equity_df["action"] != "START").sum())

                # ── Sim stats ───────────────────────────────────────────────
                s1, s2, s3 = st.columns(3)
                s1.metric("Starting Capital", fmt_usd(bt_init_cash))
                s2.metric(
                    "Final Portfolio Value",
                    fmt_usd(final_val),
                    delta       = f"{total_return_pct:+.2f}%",
                    delta_color = "normal",
                )
                s3.metric("Number of Trades", n_trades)

                st.divider()

                # ── Equity curve ────────────────────────────────────────────
                st.subheader("Equity Curve")
                st.plotly_chart(_build_equity_chart(equity_df), use_container_width=True)

                st.divider()

                # ── Backtest trade history ───────────────────────────────────
                st.subheader("Trade History")
                with get_connection(BACKTEST_DB_PATH) as _bt_conn:
                    bt_trades = [
                        dict(r) for r in _bt_conn.execute(
                            """
                            SELECT timestamp, ticker, type, price,
                                   quantity, sentiment_score
                            FROM   trades
                            ORDER  BY id DESC
                            LIMIT  50
                            """
                        ).fetchall()
                    ]
                if bt_trades:
                    bt_df = pd.DataFrame(bt_trades)
                    bt_df["timestamp"] = (
                        pd.to_datetime(bt_df["timestamp"]).dt.strftime("%m-%d %H:%M")
                    )
                    bt_df = bt_df.rename(columns={
                        "timestamp"      : "Time",
                        "ticker"         : "Ticker",
                        "type"           : "Type",
                        "quantity"       : "Qty",
                        "price"          : "Price",
                        "sentiment_score": "Sentiment",
                    })
                    st.dataframe(
                        bt_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Type"     : st.column_config.TextColumn(width="small"),
                            "Price"    : st.column_config.NumberColumn(format="$%.2f"),
                            "Qty"      : st.column_config.NumberColumn(format="%.4g"),
                            "Sentiment": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )
                else:
                    st.caption("No trades in backtest database.")

        except Exception as exc:
            st.error(f"Failed to load backtest data: {exc}")

    st.stop()  # Skip live-mode content below

# ── Live mode ───────────────────────────────────────────────────────────────
engine    = _engine()
portfolio = engine.get_portfolio()
init_cash = _initial_cash()

# ── Header ─────────────────────────────────────────────────────────────────
hcol, pcol = st.columns([3, 1])
with hcol:
    st.header(f"{ticker} — Market Dashboard")
live_price: Optional[float] = _price(ticker)
with pcol:
    if live_price:
        st.metric("Live price", fmt_usd(live_price))
    else:
        st.warning("Price unavailable")

# ── Portfolio metrics ──────────────────────────────────────────────────────
st.subheader("Portfolio")

total_assets: float = sum(
    (_price(p.ticker) or p.average_buy_price) * p.quantity
    for p in portfolio.positions
)
total_value: float  = portfolio.cash + total_assets
total_pnl: float    = total_value - init_cash
total_pnl_pct       = (total_pnl / init_cash * 100) if init_cash else 0.0

m1, m2, m3 = st.columns(3)
m1.metric(
    "💵 Cash",
    fmt_usd(portfolio.cash),
)
m2.metric(
    "📦 Assets Value",
    fmt_usd(total_assets),
)
m3.metric(
    "💼 Total Portfolio",
    fmt_usd(total_value),
    delta      =_pnl_delta(total_pnl, total_pnl_pct),
    delta_color="normal",
)

st.divider()

# ── Price + Volume chart ────────────────────────────────────────────────────
st.subheader(f"Price Chart  ·  {ticker}  ·  {period} / {interval}")

df = _history(ticker, period, interval)

if df is not None and not df.empty:
    st.plotly_chart(_build_chart(df, ticker, chart_type), use_container_width=True)
else:
    st.warning(
        f"Could not load price data for **{ticker}**. "
        "Check the ticker symbol or your internet connection."
    )

st.divider()

# ── News feed  +  Trade history ────────────────────────────────────────────
news_col, hist_col = st.columns([1.3, 1], gap="large")

with news_col:
    st.subheader(f"📰 Latest News  ·  {ticker}")
    news_items = _news(ticker)

    if news_items:
        for item in news_items[:6]:
            with st.expander(item["title"], expanded=False):
                st.markdown(
                    f'<span class="pub-chip">{item["publisher"]}</span>',
                    unsafe_allow_html=True,
                )
                if item["link"]:
                    st.markdown(f"[Read full article →]({item['link']})")
    else:
        st.caption("No news headlines found for this ticker.")

with hist_col:
    st.subheader("📋 Trade History")
    history = engine.get_trade_history(limit=20)

    if history:
        hist_df = pd.DataFrame(history)
        hist_df["timestamp"] = (
            pd.to_datetime(hist_df["timestamp"]).dt.strftime("%m-%d %H:%M")
        )
        hist_df = hist_df[
            ["timestamp", "ticker", "type", "quantity", "price", "sentiment_score"]
        ].rename(columns={
            "timestamp"      : "Time",
            "ticker"         : "Ticker",
            "type"           : "Type",
            "quantity"       : "Qty",
            "price"          : "Price",
            "sentiment_score": "Sentiment",
        })
        st.dataframe(
            hist_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Type"     : st.column_config.TextColumn(width="small"),
                "Price"    : st.column_config.NumberColumn(format="$%.2f"),
                "Qty"      : st.column_config.NumberColumn(format="%.4g"),
                "Sentiment": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.caption("No trades recorded yet — use the form below.")

st.divider()

# ── Manual Trade ───────────────────────────────────────────────────────────
st.subheader("⚡ Manual Trade")
st.caption("Execute a paper trade to test the full engine loop end-to-end.")

with st.form("trade_form", clear_on_submit=True):
    fc1, fc2, fc3, fc4 = st.columns([0.8, 1, 1.5, 1.2])

    with fc1:
        action: str = st.selectbox("Action", ["BUY", "SELL"])
    with fc2:
        qty: float = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.5)
    with fc3:
        default_px = float(f"{live_price:.2f}") if live_price else 100.0
        trade_px: float = st.number_input(
            "Price (USD)", min_value=0.01, value=default_px, step=0.01,
            help="Pre-filled with the live price — adjust manually if needed.",
        )
    with fc4:
        st.markdown("<br>", unsafe_allow_html=True)
        icon = "🟢" if action == "BUY" else "🔴"
        submitted: bool = st.form_submit_button(
            f"{icon}  Execute {action}",
            use_container_width=True,
            type="primary",
        )

# Processed outside the form block so st.rerun() works correctly
if submitted:
    result = (
        engine.buy(ticker,  price=trade_px, quantity=qty)
        if action == "BUY"
        else engine.sell(ticker, price=trade_px, quantity=qty)
    )
    if result.success:
        st.success(f"✅  {result.message}")
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now()
        st.rerun()
    else:
        st.error(f"❌  {result.message}")
