"""Fetch government bond yields from the FRED API.

Pulls the constant-maturity Treasury yield series that correspond to the
tenors used in ``testcode.py`` (1Y, 2Y, 3Y, 5Y, 10Y, 20Y) and returns them in
the same shape ``load_yields`` produces: a DataFrame indexed by date, one
column per tenor, values in percent.

Requires a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)
set as FRED_API_KEY in a .env file (see .env.example).

Usage
-----
    python fred_data.py                       # fetch and print a preview
    python fred_data.py --start 2015-01-01
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import requests
from dotenv import load_dotenv

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs for constant-maturity Treasury yields, matching the tenors
# used elsewhere in this project.
TENOR_SERIES = {
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "20Y": "DGS20",
}


def get_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return api_key


def fetch_series(series_id: str, api_key: str, start_date: str = "2015-01-01") -> pd.Series:
    """Fetch one FRED series as a date-indexed Series of floats."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    observations = resp.json()["observations"]

    df = pd.DataFrame(observations)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    return df.set_index("date")["value"].rename(series_id)


def fetch_yield_curve(start_date: str = "2015-01-01", api_key: str | None = None) -> pd.DataFrame:
    """Fetch all tenors and combine into one DataFrame, in percent, matching load_yields()."""
    api_key = api_key or get_api_key()

    series = {
        label: fetch_series(series_id, api_key, start_date)
        for label, series_id in TENOR_SERIES.items()
    }
    df = pd.DataFrame(series)
    df = df.sort_index().dropna(how="all")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01", help="Observation start date (YYYY-MM-DD)")
    args = parser.parse_args()

    yields = fetch_yield_curve(start_date=args.start)
    print(f"Fetched {len(yields)} rows, {yields.index[0].date()} .. {yields.index[-1].date()}")
    print(f"Columns: {list(yields.columns)}")
    print(f"Missing values per tenor:\n{yields.isna().sum()}")
    print("\nPreview:")
    print(yields.tail(10))


if __name__ == "__main__":
    main()
