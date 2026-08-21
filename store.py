"""SQLite storage for the yield curve time series.

Observations are stored in long format (date, tenor, value) rather than one
column per tenor, so new tenors can be added later without a schema change
and re-fetching the same date range safely upserts instead of duplicating.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from fred_data import TENOR_SERIES

DB_PATH = Path(__file__).parent / "yields.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS yields (
    date TEXT NOT NULL,
    tenor TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (date, tenor)
);
"""

TENOR_ORDER = list(TENOR_SERIES.keys())  # curve order (1Y..20Y), not alphabetical


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    return conn


def upsert_yields(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert or update yields from a wide DataFrame (date index, tenor columns)."""
    long = df.stack().reset_index()
    long.columns = ["date", "tenor", "value"]
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")

    conn.executemany(
        "INSERT INTO yields (date, tenor, value) VALUES (?, ?, ?) "
        "ON CONFLICT(date, tenor) DO UPDATE SET value = excluded.value",
        long.itertuples(index=False, name=None),
    )
    conn.commit()
    return len(long)


def load_yields(
    conn: sqlite3.Connection, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Read back as a wide DataFrame (date index, tenor columns), curve-ordered."""
    query = "SELECT date, tenor, value FROM yields"
    clauses, params = [], []
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    long = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    if long.empty:
        return pd.DataFrame()

    wide = long.pivot(index="date", columns="tenor", values="value").sort_index()
    wide = wide.reindex(columns=[t for t in TENOR_ORDER if t in wide.columns])
    wide.index.name = "Date"
    return wide
