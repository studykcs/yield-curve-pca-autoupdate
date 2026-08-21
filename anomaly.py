"""Detect anomalous days in the yield curve using PCA factor scores.

A day is flagged when its PC1 (level), PC2 (slope) or PC3 (curvature) score
exceeds a z-score threshold, where z is the day's score divided by that
component's own standard deviation (sqrt of its eigenvalue). Because the
components are orthogonal this is just a per-factor sigma check, not a
joint statistic.

Usage
-----
    python anomaly.py --from-db
    python anomaly.py --from-db --threshold 2.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from store import get_connection, load_yields
from testcode import OUT, PCAResult, run_pca

PC_LABELS = {"PC1": "Level", "PC2": "Slope", "PC3": "Curvature"}
ALERT_LOG = OUT / "alerts.json"


def zscores(res: PCAResult, n_components: int = 3) -> pd.DataFrame:
    cols = [f"PC{i + 1}" for i in range(n_components)]
    vol = res.factor_vol_bp[:n_components]
    return res.scores[cols] / vol


def detect_anomalies(res: PCAResult, threshold: float = 3.0, n_components: int = 3) -> pd.DataFrame:
    """One row per (date, component) breach, sorted by date."""
    z = zscores(res, n_components)
    breached = z.abs() > threshold

    rows = [
        {
            "date": date,
            "component": pc,
            "label": PC_LABELS.get(pc, pc),
            "z_score": z.loc[date, pc],
            "move_bp": res.scores.loc[date, pc],
        }
        for date in z.index[breached.any(axis=1)]
        for pc in z.columns
        if breached.loc[date, pc]
    ]
    columns = ["date", "component", "label", "z_score", "move_bp"]
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def alert_new(anomalies: pd.DataFrame, log_path: Path = ALERT_LOG) -> pd.DataFrame:
    """Print and persist only the anomalies not already in the alert log.

    Re-running the pipeline recomputes the same historical anomalies every
    time, so without dedup every run would re-alert on old news. The log
    tracks (date, component) pairs already surfaced.
    """
    anomalies = anomalies.copy()
    anomalies["date"] = pd.to_datetime(anomalies["date"]).dt.strftime("%Y-%m-%d")

    seen = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    seen_keys = {(r["date"], r["component"]) for r in seen}

    is_new = ~anomalies.apply(lambda r: (r["date"], r["component"]) in seen_keys, axis=1)
    new = anomalies[is_new]

    if new.empty:
        print("No new anomalies since last check.")
        return new

    print(f"\n{len(new)} NEW anomal{'y' if len(new) == 1 else 'ies'} since last check:\n")
    for _, r in new.iterrows():
        print(f"  ALERT  {r['date']}  {r['label']:<10} z={r['z_score']:+.2f}  move={r['move_bp']:+.2f} bp")

    new_records = new[["date", "component", "label", "z_score", "move_bp"]].to_dict("records")
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    for rec in new_records:
        rec["detected_at"] = now

    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(seen + new_records, indent=2), encoding="utf-8")
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Earliest date to include (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Z-score threshold (default 3.0)")
    parser.add_argument("--reset-log", action="store_true", help="Clear the alert log before checking")
    args = parser.parse_args()

    if args.reset_log and ALERT_LOG.exists():
        ALERT_LOG.unlink()

    conn = get_connection()
    yields = load_yields(conn, start=args.start)
    conn.close()
    if yields.empty:
        raise SystemExit("No data in yields.db. Run collect.py first.")

    res = run_pca(yields)
    anomalies = detect_anomalies(res, threshold=args.threshold)

    if anomalies.empty:
        print(f"No anomalies above {args.threshold}-sigma in {len(res.scores)} days.")
        return

    print(f"{len(anomalies)} anomalies above {args.threshold}-sigma in this window.")
    alert_new(anomalies)


if __name__ == "__main__":
    main()
