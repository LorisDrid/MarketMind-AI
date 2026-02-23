"""
engine.py - Virtual broker / portfolio engine.

Manages the simulated wallet, executes paper trades, and persists all
state to SQLite via database.py.  Designed so that a Reinforcement
Learning agent can replace (or extend) the strategy layer without
touching this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from database import get_connection, init_db

logger = logging.getLogger(__name__)

# Flat fee (USD) charged on every BUY and SELL execution.
# Simulates realistic broker commissions / spread costs.
TRANSACTION_FEE: float = 1.0


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeResult:
    """Outcome of a :py:meth:`Engine.buy` or :py:meth:`Engine.sell` call."""

    success: bool
    ticker: str
    type: str  # 'BUY' | 'SELL'
    quantity: float
    price: float
    total_value: float
    cash_after: float
    message: str
    sentiment_score: Optional[float] = None
    strategy_name:   Optional[str]   = None


@dataclass
class Position:
    ticker: str
    quantity: float
    average_buy_price: float

    @property
    def market_value(self) -> float:
        """Placeholder — real value requires a live/cached price lookup."""
        return self.quantity * self.average_buy_price


@dataclass
class PortfolioSnapshot:
    cash: float
    positions: list[Position] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_invested(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def total_value(self) -> float:
        return self.cash + self.total_invested


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Paper-trading broker.

    All monetary values are in USD. Prices passed in are assumed to be the
    current market price for the given ticker (fetched externally by
    data_manager.py or the strategy layer).

    Args:
        starting_cash: Initial virtual balance (used only on first run).
        db_path: Override for the SQLite file path (mainly for testing).
    """

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        db_path=None,
        strategy_name: Optional[str] = None,
    ) -> None:
        kwargs = {"starting_cash": starting_cash}
        if db_path is not None:
            kwargs["db_path"] = db_path
        init_db(**kwargs)
        self._db_path       = db_path        # None → use module default
        self._strategy_name = strategy_name  # tagged on every persisted trade

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self):
        """Return a new database connection (caller manages lifecycle)."""
        if self._db_path is not None:
            return get_connection(self._db_path)
        return get_connection()

    def _get_cash(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT current_cash FROM portfolio ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("Portfolio table is empty — was init_db() called?")
        return float(row["current_cash"])

    def _set_cash(self, conn, new_balance: float) -> None:
        conn.execute(
            "INSERT INTO portfolio (current_cash) VALUES (?)", (new_balance,)
        )

    def _get_position(self, conn, ticker: str) -> Optional[Position]:
        row = conn.execute(
            "SELECT ticker, quantity, average_buy_price FROM positions WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()
        if row is None:
            return None
        return Position(
            ticker=row["ticker"],
            quantity=float(row["quantity"]),
            average_buy_price=float(row["average_buy_price"]),
        )

    def _log_trade(
        self,
        conn,
        ticker: str,
        trade_type: str,
        price: float,
        quantity: float,
        sentiment_score: Optional[float],
    ) -> None:
        conn.execute(
            """
            INSERT INTO trades
                (ticker, type, price, quantity, sentiment_score, strategy_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker.upper(), trade_type, price, quantity,
             sentiment_score, self._strategy_name),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def buy(
        self,
        ticker: str,
        price: float,
        quantity: float,
        sentiment_score: Optional[float] = None,
    ) -> TradeResult:
        """Purchase *quantity* shares of *ticker* at *price*.

        Args:
            ticker: Stock symbol (e.g. ``"AAPL"``).
            price: Current market price per share in USD.
            quantity: Number of shares to buy (fractional shares supported).
            sentiment_score: Optional AI sentiment in ``[-1.0, 1.0]``.

        Returns:
            :class:`TradeResult` describing the outcome.
        """
        ticker = ticker.upper()
        total_cost = price * quantity

        if price <= 0 or quantity <= 0:
            msg = f"BUY {ticker}: price and quantity must be positive."
            logger.warning(msg)
            return TradeResult(
                success=False, ticker=ticker, type="BUY",
                quantity=quantity, price=price, total_value=total_cost,
                cash_after=self._get_cash(), message=msg,
                sentiment_score=sentiment_score,
            )

        with self._conn() as conn:
            cash = float(
                conn.execute(
                    "SELECT current_cash FROM portfolio ORDER BY id DESC LIMIT 1"
                ).fetchone()["current_cash"]
            )

            if total_cost + TRANSACTION_FEE > cash:
                msg = (
                    f"BUY {ticker}: insufficient funds "
                    f"(need ${total_cost + TRANSACTION_FEE:.2f} incl. fee, "
                    f"have ${cash:.2f})."
                )
                logger.warning(msg)
                return TradeResult(
                    success=False, ticker=ticker, type="BUY",
                    quantity=quantity, price=price, total_value=total_cost,
                    cash_after=cash, message=msg,
                    sentiment_score=sentiment_score,
                )

            new_cash = cash - total_cost - TRANSACTION_FEE
            self._set_cash(conn, new_cash)

            existing = self._get_position(conn, ticker)
            if existing is None:
                conn.execute(
                    "INSERT INTO positions (ticker, quantity, average_buy_price) "
                    "VALUES (?, ?, ?)",
                    (ticker, quantity, price),
                )
            else:
                # Update weighted-average cost basis
                total_qty = existing.quantity + quantity
                avg_price = (
                    existing.quantity * existing.average_buy_price + quantity * price
                ) / total_qty
                conn.execute(
                    "UPDATE positions SET quantity = ?, average_buy_price = ? "
                    "WHERE ticker = ?",
                    (total_qty, avg_price, ticker),
                )

            self._log_trade(conn, ticker, "BUY", price, quantity, sentiment_score)
            conn.commit()

        msg = (
            f"BUY {quantity:.4f} {ticker} @ ${price:.4f} "
            f"(total ${total_cost:.2f} + ${TRANSACTION_FEE:.2f} fee) "
            f"→ cash ${new_cash:.2f}"
        )
        logger.info(msg)
        return TradeResult(
            success=True, ticker=ticker, type="BUY",
            quantity=quantity, price=price, total_value=total_cost,
            cash_after=new_cash, message=msg,
            sentiment_score=sentiment_score,
            strategy_name=self._strategy_name,
        )

    def sell(
        self,
        ticker: str,
        price: float,
        quantity: float,
        sentiment_score: Optional[float] = None,
    ) -> TradeResult:
        """Sell *quantity* shares of *ticker* at *price*.

        Args:
            ticker: Stock symbol.
            price: Current market price per share in USD.
            quantity: Number of shares to sell.  Pass ``None`` (or omit) to
                sell the entire position (handled at strategy layer).
            sentiment_score: Optional AI sentiment in ``[-1.0, 1.0]``.

        Returns:
            :class:`TradeResult` describing the outcome.
        """
        ticker = ticker.upper()
        total_proceeds = price * quantity

        if price <= 0 or quantity <= 0:
            msg = f"SELL {ticker}: price and quantity must be positive."
            logger.warning(msg)
            return TradeResult(
                success=False, ticker=ticker, type="SELL",
                quantity=quantity, price=price, total_value=total_proceeds,
                cash_after=self._get_cash(), message=msg,
                sentiment_score=sentiment_score,
            )

        with self._conn() as conn:
            existing = self._get_position(conn, ticker)

            if existing is None or existing.quantity < quantity:
                held = existing.quantity if existing else 0.0
                msg = (
                    f"SELL {ticker}: insufficient shares "
                    f"(need {quantity:.4f}, hold {held:.4f})."
                )
                logger.warning(msg)
                cash = float(
                    conn.execute(
                        "SELECT current_cash FROM portfolio ORDER BY id DESC LIMIT 1"
                    ).fetchone()["current_cash"]
                )
                return TradeResult(
                    success=False, ticker=ticker, type="SELL",
                    quantity=quantity, price=price, total_value=total_proceeds,
                    cash_after=cash, message=msg,
                    sentiment_score=sentiment_score,
                )

            cash = float(
                conn.execute(
                    "SELECT current_cash FROM portfolio ORDER BY id DESC LIMIT 1"
                ).fetchone()["current_cash"]
            )
            new_cash = cash + total_proceeds - TRANSACTION_FEE
            self._set_cash(conn, new_cash)

            remaining_qty = existing.quantity - quantity
            if remaining_qty < 1e-9:  # Treat as fully closed
                conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
            else:
                conn.execute(
                    "UPDATE positions SET quantity = ? WHERE ticker = ?",
                    (remaining_qty, ticker),
                )

            self._log_trade(conn, ticker, "SELL", price, quantity, sentiment_score)
            conn.commit()

        msg = (
            f"SELL {quantity:.4f} {ticker} @ ${price:.4f} "
            f"(proceeds ${total_proceeds:.2f} − ${TRANSACTION_FEE:.2f} fee) "
            f"→ cash ${new_cash:.2f}"
        )
        logger.info(msg)
        return TradeResult(
            success=True, ticker=ticker, type="SELL",
            quantity=quantity, price=price, total_value=total_proceeds,
            cash_after=new_cash, message=msg,
            sentiment_score=sentiment_score,
            strategy_name=self._strategy_name,
        )

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def get_portfolio(self) -> PortfolioSnapshot:
        """Return a snapshot of the current portfolio state."""
        with self._conn() as conn:
            cash_row = conn.execute(
                "SELECT current_cash FROM portfolio ORDER BY id DESC LIMIT 1"
            ).fetchone()
            cash = float(cash_row["current_cash"]) if cash_row else 0.0

            pos_rows = conn.execute(
                "SELECT ticker, quantity, average_buy_price FROM positions"
            ).fetchall()

        positions = [
            Position(
                ticker=r["ticker"],
                quantity=float(r["quantity"]),
                average_buy_price=float(r["average_buy_price"]),
            )
            for r in pos_rows
        ]
        return PortfolioSnapshot(cash=cash, positions=positions)

    def get_trade_history(self, limit: int = 100) -> list[dict]:
        """Return the most recent *limit* trades as plain dicts."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, ticker, type, price, quantity, sentiment_score
                FROM trades
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cash(self) -> float:
        """Return current cash balance."""
        return self._get_cash()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = Engine(starting_cash=10_000.0)
    snap = engine.get_portfolio()
    logger.info("Initial portfolio: cash=$%.2f, positions=%d", snap.cash, len(snap.positions))

    r1 = engine.buy("AAPL", price=190.50, quantity=10)
    logger.info("After buy:  %s", r1.message)

    r2 = engine.sell("AAPL", price=195.00, quantity=5)
    logger.info("After sell: %s", r2.message)

    snap2 = engine.get_portfolio()
    logger.info(
        "Final portfolio: cash=$%.2f, positions=%s",
        snap2.cash,
        [(p.ticker, p.quantity) for p in snap2.positions],
    )
