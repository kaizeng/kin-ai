"""SQLite cache for price and dividend data."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "kin_ai.db"

# Data older than this is considered stale and re-downloaded
CACHE_EXPIRY_HOURS = 12


def _connect() -> sqlite3.Connection:
    """Open (and optionally create) the database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker   TEXT    NOT NULL,
            date     TEXT    NOT NULL,
            close    REAL    NOT NULL,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS dividends (
            ticker   TEXT    NOT NULL,
            date     TEXT    NOT NULL,
            amount   REAL    NOT NULL,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS cache_meta (
            ticker     TEXT    NOT NULL,
            data_type  TEXT    NOT NULL,   -- 'prices' or 'dividends'
            start_date TEXT    NOT NULL,
            end_date   TEXT    NOT NULL,
            updated_at TEXT    NOT NULL,
            PRIMARY KEY (ticker, data_type)
        );

        CREATE TABLE IF NOT EXISTS universe (
            ticker      TEXT PRIMARY KEY,
            name        TEXT,
            asset_class TEXT,
            sector      TEXT,
            description TEXT
        );
    """)
    conn.close()


def _is_fresh(
    conn: sqlite3.Connection,
    ticker: str,
    data_type: str,
    start: str,
    end: str,
) -> bool:
    """Check whether cached data covers the requested range and is recent."""
    row = conn.execute(
        "SELECT start_date, end_date, updated_at FROM cache_meta "
        "WHERE ticker = ? AND data_type = ?",
        (ticker, data_type),
    ).fetchone()
    if row is None:
        return False
    cached_start, cached_end, updated_at = row
    if cached_start > start or cached_end < end:
        return False
    age = datetime.utcnow() - datetime.fromisoformat(updated_at)
    return age < timedelta(hours=CACHE_EXPIRY_HOURS)


# ── Prices ──────────────────────────────────────────────────────

def read_prices(
    tickers: List[str], start: str, end: str
) -> Optional[pd.DataFrame]:
    """Return cached prices if ALL tickers are fresh, else None."""
    init_db()
    conn = _connect()
    try:
        for t in tickers:
            if not _is_fresh(conn, t, "prices", start, end):
                return None
        df = pd.read_sql_query(
            f"SELECT date, ticker, close FROM prices "
            f"WHERE ticker IN ({','.join('?' * len(tickers))}) "
            f"AND date >= ? AND date <= ? "
            f"ORDER BY date",
            conn,
            params=[*tickers, start, end],
        )
    finally:
        conn.close()

    if df.empty:
        return None
    prices = df.pivot(index="date", columns="ticker", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    return prices[tickers]  # preserve column order


def write_prices(prices: pd.DataFrame, tickers: List[str]) -> None:
    """Upsert price rows and update cache_meta."""
    init_db()
    conn = _connect()
    try:
        for t in tickers:
            if t not in prices.columns:
                continue
            series = prices[t].dropna()
            rows = [(t, d.strftime("%Y-%m-%d"), float(v)) for d, v in series.items()]
            conn.executemany(
                "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                rows,
            )
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta "
                "(ticker, data_type, start_date, end_date, updated_at) "
                "VALUES (?, 'prices', ?, ?, ?)",
                (
                    t,
                    series.index.min().strftime("%Y-%m-%d"),
                    series.index.max().strftime("%Y-%m-%d"),
                    datetime.utcnow().isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ── Dividends ───────────────────────────────────────────────────

def read_dividends(
    tickers: List[str], start: str, end: str
) -> Optional[pd.DataFrame]:
    """Return cached dividends if ALL tickers are fresh, else None."""
    init_db()
    conn = _connect()
    try:
        for t in tickers:
            if not _is_fresh(conn, t, "dividends", start, end):
                return None
        df = pd.read_sql_query(
            f"SELECT date, ticker, amount FROM dividends "
            f"WHERE ticker IN ({','.join('?' * len(tickers))}) "
            f"AND date >= ? AND date <= ? "
            f"ORDER BY date",
            conn,
            params=[*tickers, start, end],
        )
    finally:
        conn.close()

    if df.empty:
        # All tickers cached but no dividend events → return zeros
        return pd.DataFrame()
    divs = df.pivot(index="date", columns="ticker", values="amount").fillna(0.0)
    divs.index = pd.to_datetime(divs.index)
    divs.index.name = "Date"
    # Ensure all requested tickers have a column
    for t in tickers:
        if t not in divs.columns:
            divs[t] = 0.0
    return divs[tickers]


def write_dividends(
    dividends: pd.DataFrame, tickers: List[str], start: str, end: str
) -> None:
    """Upsert dividend rows and update cache_meta."""
    init_db()
    conn = _connect()
    try:
        for t in tickers:
            # Write actual dividend events
            if t in dividends.columns:
                series = dividends[t]
                nonzero = series[series > 0].dropna()
                rows = [
                    (t, d.strftime("%Y-%m-%d"), float(v))
                    for d, v in nonzero.items()
                ]
                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO dividends (ticker, date, amount) "
                        "VALUES (?, ?, ?)",
                        rows,
                    )
            # Always update meta so we know we checked this ticker
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta "
                "(ticker, data_type, start_date, end_date, updated_at) "
                "VALUES (?, 'dividends', ?, ?, ?)",
                (t, start, end, datetime.utcnow().isoformat()),
            )
        conn.commit()
    finally:
        conn.close()


# ── Universe ────────────────────────────────────────────────────

def write_universe(records: List[dict]) -> None:
    """Insert/update ETF universe rows."""
    init_db()
    conn = _connect()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO universe "
            "(ticker, name, asset_class, sector, description) "
            "VALUES (:ticker, :name, :asset_class, :sector, :description)",
            records,
        )
        conn.commit()
    finally:
        conn.close()


def read_universe() -> pd.DataFrame:
    """Read the full ETF universe from the database."""
    init_db()
    conn = _connect()
    try:
        df = pd.read_sql_query("SELECT * FROM universe ORDER BY asset_class, ticker", conn)
    finally:
        conn.close()
    return df
