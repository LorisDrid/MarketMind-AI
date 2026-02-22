"""
app.py — MarketMind AI · Streamlit Trading Dashboard

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_manager import get_historical_data, get_latest_price, get_ticker_news
from engine import Engine

# ── Page config (must be the very first Streamlit call) ───────────────────
st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "MarketMind AI — Paper Trading Simulator"},
)

logging.basicConfig(level=logging.WARNING)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Tighten top padding */
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

        /* Metric cards */
        [data-testid="metric-container"] {
            background  : rgba(255, 255, 255, 0.03);
            border      : 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.6rem;
            padding     : 0.9rem 1.1rem;
        }

        /* Section headers */
        h2 { margin-top: 0.25rem !important; }

        /* News publisher label */
        .publisher-tag {
            font-size     : 0.72rem;
            font-weight   : 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color         : #8a8a8a;
            margin-bottom : 0.4rem;
        }

        /* Trade type badges */
        .badge-buy  { color: #2ecc71; font-weight: 700; }
        .badge-sell { color: #e74c3c; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────
TICKERS: list[str] = ["NVDA", "AAPL", "MSFT", "TSLA"]

# Smart interval defaults: keep bar count reasonable across periods
_PERIOD_INTERVAL: dict[str, str] = {
    "1d" : "5m",
    "5d" : "30m",
    "30d": "1h",
    "60d": "1h",
}


# ── Cached helpers (Streamlit layer) ───────────────────────────────────────

@st.cache_resource
def _engine() -> Engine:
    """Single Engine instance for the whole session."""
    return Engine()


@st.cache_data(ttl=300, show_spinner="Fetching price history…")
def _history(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    return get_historical_data(ticker, period=period, interval=interval)


@st.cache_data(ttl=60, show_spinner=False)
def _price(ticker: str) -> Optional[float]:
    return get_latest_price(ticker)


@st.cache_data(ttl=600, show_spinner="Fetching news…")
def _news(ticker: str) -> list[dict]:
    return get_ticker_news(ticker)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 MarketMind AI")
    st.caption("Paper Trading Simulator · MVP")
    st.divider()

    ticker: str = st.selectbox("Ticker", TICKERS, index=0)
    period: str = st.selectbox("Period", list(_PERIOD_INTERVAL.keys()), index=1)
    interval: str = _PERIOD_INTERVAL[period]
    st.caption(f"Auto interval: `{interval}`")

    chart_type: str = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

    if st.button("🔄  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # ── Open positions (sidebar) ───────────────────────────────────────────
    st.subheader("Open Positions")
    engine = _engine()
    portfolio = engine.get_portfolio()

    if portfolio.positions:
        for pos in portfolio.positions:
            live = _price(pos.ticker)
            if live:
                live_val  = live * pos.quantity
                cost_val  = pos.average_buy_price * pos.quantity
                pnl       = live_val - cost_val
                pnl_pct   = (pnl / cost_val) * 100 if cost_val else 0
                delta_str = f"{'▲' if pnl >= 0 else '▼'} ${abs(pnl):,.2f} ({pnl_pct:+.1f}%)"
                st.metric(
                    label      =f"{pos.ticker}  ·  {pos.quantity:g} sh",
                    value      =f"${live_val:,.2f}",
                    delta      =delta_str,
                    delta_color="normal",
                )
            else:
                st.metric(
                    label=f"{pos.ticker}  ·  {pos.quantity:g} sh",
                    value=f"${pos.market_value:,.2f}",
                    help ="Live price unavailable — showing cost-basis value.",
                )
    else:
        st.caption("No open positions.")


# ── Main content ───────────────────────────────────────────────────────────
engine = _engine()
portfolio = engine.get_portfolio()

# Page header + live price
hcol, pcol = st.columns([3, 1])
with hcol:
    st.header(f"{ticker} — Market Dashboard")
live_price = _price(ticker)
with pcol:
    if live_price:
        st.metric("Live price", f"${live_price:,.2f}")
    else:
        st.warning("Price unavailable")

# ── Portfolio metrics row ──────────────────────────────────────────────────
st.subheader("Portfolio")

# Recompute total assets with live prices where possible
total_assets: float = sum(
    (_price(p.ticker) or p.average_buy_price) * p.quantity
    for p in portfolio.positions
)
total_value: float = portfolio.cash + total_assets

m1, m2, m3 = st.columns(3)
m1.metric("💵 Cash",            f"${portfolio.cash:,.2f}")
m2.metric("📦 Assets Value",    f"${total_assets:,.2f}")
m3.metric("💼 Total Portfolio", f"${total_value:,.2f}")

st.divider()

# ── Candlestick / line chart ───────────────────────────────────────────────
st.subheader(f"Price Chart · {ticker}")

df = _history(ticker, period, interval)

if df is not None and not df.empty:
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x    =df.index,
                open =df["Open"],
                high =df["High"],
                low  =df["Low"],
                close=df["Close"],
                name =ticker,
                increasing_line_color="#2ecc71",
                decreasing_line_color="#e74c3c",
                increasing_fillcolor ="#2ecc71",
                decreasing_fillcolor ="#e74c3c",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x         =df.index,
                y         =df["Close"],
                mode      ="lines",
                name      =ticker,
                line      =dict(color="#4e8df5", width=1.8),
                fill      ="tozeroy",
                fillcolor ="rgba(78, 141, 245, 0.07)",
            )
        )

    fig.update_layout(
        template              ="plotly_dark",
        height                =430,
        margin                =dict(l=0, r=0, t=28, b=0),
        xaxis_rangeslider_visible=False,
        xaxis =dict(showgrid=False, color="#666"),
        yaxis =dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#666"),
        paper_bgcolor         ="rgba(0,0,0,0)",
        plot_bgcolor          ="rgba(0,0,0,0)",
        hovermode             ="x unified",
        hoverlabel            =dict(bgcolor="#1e1e1e", bordercolor="#333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(f"Could not load price data for **{ticker}**. Check the ticker or your connection.")

st.divider()

# ── News feed  +  Trade history (side by side) ────────────────────────────
news_col, hist_col = st.columns([1.3, 1], gap="large")

with news_col:
    st.subheader(f"📰 Latest News · {ticker}")
    news_items = _news(ticker)

    if news_items:
        for item in news_items[:6]:
            with st.expander(item["title"], expanded=False):
                st.markdown(
                    f'<p class="publisher-tag">{item["publisher"]}</p>',
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

# ── Manual trade form ──────────────────────────────────────────────────────
st.subheader("⚡ Manual Trade")
st.caption("Execute a paper trade against the live engine to test the full loop.")

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
            help="Pre-filled with the live price — adjust if needed.",
        )
    with fc4:
        st.markdown("<br>", unsafe_allow_html=True)   # vertical align
        submitted: bool = st.form_submit_button(
            f"{'🟢' if action == 'BUY' else '🔴'}  Execute {action}",
            use_container_width=True,
            type="primary",
        )

if submitted:
    result = (
        engine.buy(ticker, price=trade_px, quantity=qty)
        if action == "BUY"
        else engine.sell(ticker, price=trade_px, quantity=qty)
    )
    if result.success:
        st.success(f"✅  {result.message}")
        st.cache_data.clear()
        st.rerun()
    else:
        st.error(f"❌  {result.message}")
