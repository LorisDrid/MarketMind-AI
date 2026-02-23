"""
database.py - SQLite schema initialization and connection management.

Provides a single source of truth for the database schema and a
thread-safe connection factory used by all other modules.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "marketmind.db"

_SCHEMA_SQL = """
-- Current cash balance and a timestamp for historical snapshots
CREATE TABLE IF NOT EXISTS portfolio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    current_cash    REAL    NOT NULL,
    timestamp       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Open positions: one row per ticker currently held
CREATE TABLE IF NOT EXISTS positions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL UNIQUE,
    quantity         REAL    NOT NULL,
    average_buy_price REAL   NOT NULL
);

-- Full transaction ledger
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ticker          TEXT    NOT NULL,
    type            TEXT    NOT NULL CHECK (type IN ('BUY', 'SELL')),
    price           REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    sentiment_score REAL,            -- NULL when trade is not AI-driven
    strategy_name   TEXT             -- profile label (e.g. Prudent / Balanced / Aggressive)
);
"""


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Return a new SQLite connection with foreign keys enabled.

    Callers are responsible for committing / closing the connection.
    Using :func:`contextlib.closing` or a ``with`` block is recommended.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Allow column access by name
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")  # Better concurrency
    return conn


def init_db(
    starting_cash: float = 10_000.0,
    db_path: Path = DB_PATH,
) -> None:
    """Create tables and seed the portfolio row if the database is new.

    Safe to call on every application start — uses CREATE TABLE IF NOT EXISTS.

    Args:
        starting_cash: Virtual wallet balance for a fresh database.
        db_path: Path to the SQLite file (default: data/marketmind.db).
    """
    logger.info("Initialising database at %s", db_path)
    with get_connection(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)

        # ── Migrate: add strategy_name if upgrading an existing DB ─────────
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()
        }
        if "strategy_name" not in existing_cols:
            conn.execute("ALTER TABLE trades ADD COLUMN strategy_name TEXT")
            logger.info("Migrated trades table: added strategy_name column.")

        # Seed portfolio only when completely empty (first run)
        row_count: int = conn.execute("SELECT COUNT(*) FROM portfolio").fetchone()[0]
        if row_count == 0:
            conn.execute(
                "INSERT INTO portfolio (current_cash) VALUES (?)",
                (starting_cash,),
            )
            logger.info("Seeded portfolio with starting cash: $%.2f", starting_cash)

        conn.commit()
    logger.info("Database ready.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    init_db()
