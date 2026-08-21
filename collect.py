"""Fetch Treasury yields from FRED and upsert them into SQLite.

Usage
-----
    python collect.py                     # fetch since 2015-01-01, store in yields.db
    python collect.py --start 2024-01-01
"""

from __future__ import annotations

import argparse

from fred_data import fetch_yield_curve
from store import DB_PATH, get_connection, upsert_yields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01", help="Observation start date (YYYY-MM-DD)")
    args = parser.parse_args()

    yields = fetch_yield_curve(start_date=args.start)

    conn = get_connection()
    n_rows = upsert_yields(conn, yields)
    conn.close()

    print(f"Upserted {n_rows} (date, tenor) rows into {DB_PATH}")
    print(f"Range: {yields.index[0].date()} .. {yields.index[-1].date()}")


if __name__ == "__main__":
    main()
