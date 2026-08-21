"""Principal component analysis of yield curve movements.

Decomposes daily changes in the government bond yield curve into orthogonal
factors.  The first three components are conventionally read as level, slope
and curvature, and typically explain well over 95% of the daily variance.

PCA is run on *changes* in yield rather than levels: yields are highly
persistent and near non-stationary, so the covariance of levels is dominated by
the trend rather than by the co-movement of interest.

Usage
-----
    python yield_curve_pca.py                       # sample data
    python yield_curve_pca.py --input yields.csv    # your own data

The input file needs a date column followed by one column per tenor, with
yields quoted in percent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_yields(path: str | Path) -> pd.DataFrame:
    """Read a yield history and return it indexed by date, in percent."""
    path = Path(path)
    reader = pd.read_excel if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv
    df = reader(path)

    df.columns = df.columns.str.strip()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def sample_yields(n_days: int = 1250, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic curve history with a realistic factor structure.

    Three latent factors are imposed with loadings shaped like level, slope and
    curvature, plus idiosyncratic noise per tenor.  This exists so the script
    is runnable without redistributing licensed market data; swap in a real
    file with ``--input``.
    """
    rng = np.random.default_rng(seed)
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
    labels = ["1Y", "2Y", "3Y", "5Y", "10Y", "20Y"]

    level = np.ones_like(tenors)
    slope = np.tanh((tenors - 5.0) / 5.0)
    curvature = 1.0 - 2.0 * np.exp(-((np.log(tenors) - np.log(4.0)) ** 2) / 0.5)
    loadings = np.column_stack([level, slope, curvature])

    factor_vol = np.array([5.5, 2.6, 1.1])  # bp per day
    factors = rng.standard_normal((n_days, 3)) * factor_vol
    noise = rng.standard_normal((n_days, len(tenors))) * 0.45

    changes_bp = factors @ loadings.T + noise
    start = np.array([3.20, 3.28, 3.35, 3.48, 3.62, 3.70])
    levels = start + np.cumsum(changes_bp, axis=0) / 100.0

    dates = pd.bdate_range("2021-01-04", periods=n_days, name="Date")
    return pd.DataFrame(levels, index=dates, columns=labels)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
@dataclass
class PCAResult:
    """Eigen-decomposition of the covariance of daily yield changes."""

    changes_bp: pd.DataFrame
    cov: pd.DataFrame
    eigenvalues: np.ndarray
    loadings: pd.DataFrame
    scores: pd.DataFrame

    @property
    def explained(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.explained)

    @property
    def factor_vol_bp(self) -> np.ndarray:
        """Daily volatility attributable to each component, in basis points."""
        return np.sqrt(self.eigenvalues)


def run_pca(yields_pct: pd.DataFrame, n_components: int = 3) -> PCAResult:
    """Covariance PCA on daily changes, in basis points."""
    changes = (yields_pct * 100).diff().dropna()

    cov = changes.cov()
    eigenvalues, eigenvectors = np.linalg.eigh(cov.to_numpy())

    # eigh returns ascending order; flip to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Sign convention: make the first loading of each component positive so
    # repeated runs and different samples stay comparable.
    signs = np.sign(eigenvectors[0, :])
    signs[signs == 0] = 1.0
    eigenvectors = eigenvectors * signs

    names = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    loadings = pd.DataFrame(eigenvectors, index=changes.columns, columns=names)
    scores = pd.DataFrame(
        changes.to_numpy() @ eigenvectors[:, :n_components],
        index=changes.index,
        columns=names[:n_components],
    )

    return PCAResult(changes, cov, eigenvalues, loadings, scores)


def reconstruction_r2(res: PCAResult, n_components: int = 3) -> pd.Series:
    """Share of each tenor's variance captured by the first k components.

    Because the components are orthogonal, this is just the sum of squared
    loadings weighted by eigenvalue, divided by the tenor's own variance.
    """
    v = res.loadings.to_numpy()[:, :n_components]
    lam = res.eigenvalues[:n_components]
    explained = (v**2 * lam).sum(axis=1)
    total = np.diag(res.cov.to_numpy())
    return pd.Series(explained / total, index=res.cov.index, name=f"R2_PC1-{n_components}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(res: PCAResult) -> None:
    print("Daily yield changes (bp)")
    print(f"  observations : {len(res.changes_bp)}")
    print(f"  tenors       : {list(res.changes_bp.columns)}")
    print(f"  period       : {res.changes_bp.index[0].date()} .. {res.changes_bp.index[-1].date()}")

    print("\nExplained variance")
    for i, (ratio, cum, vol) in enumerate(
        zip(res.explained, res.cumulative, res.factor_vol_bp), start=1
    ):
        print(f"  PC{i}: {ratio * 100:6.2f}%   cumulative {cum * 100:6.2f}%   sd {vol:5.2f} bp/day")

    print("\nLoadings")
    print(res.loadings.iloc[:, :3].round(4).to_string())

    print("\nInterpretation")
    l1, l2, l3 = (res.loadings[c].to_numpy() for c in ["PC1", "PC2", "PC3"])
    print(f"  PC1 same sign across tenors    : {bool(np.all(np.sign(l1) == np.sign(l1[0])))}  -> level")
    print(f"  PC2 sign flips short vs long   : {bool(np.sign(l2[0]) != np.sign(l2[-1]))}  -> slope")
    print(f"  PC3 belly opposite to the ends : {bool(np.sign(l3[len(l3) // 2]) != np.sign(l3[0]))}  -> curvature")

    print("\nVariance of each tenor explained by PC1-3")
    print((reconstruction_r2(res) * 100).round(2).to_string())


def make_plots(res: PCAResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    names = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = [f"PC{i + 1}" for i in range(len(res.explained))]
    axes[0].bar(labels, res.explained * 100, alpha=0.75)
    axes[0].step(labels, res.cumulative * 100, where="mid", marker="o", color="black", lw=1.4)
    axes[0].set_title("Explained variance")
    axes[0].set_ylabel("% of total variance")
    axes[0].grid(alpha=0.3, ls="--", axis="y")

    for i, name in enumerate(names):
        axes[1].plot(res.loadings.index, res.loadings.iloc[:, i], marker="o", lw=2, label=name)
    axes[1].axhline(0, color="black", ls="--", alpha=0.5, lw=1)
    axes[1].set_title("Loadings by tenor")
    axes[1].set_xlabel("Tenor")
    axes[1].set_ylabel("Loading")
    axes[1].legend()
    axes[1].grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance_and_loadings.png", dpi=140)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, col, name in zip(axes, res.scores.columns, names):
        ax.plot(res.scores.index, res.scores[col], lw=1.2)
        ax.axhline(0, color="black", ls="--", alpha=0.3, lw=1)
        ax.set_title(name)
        ax.grid(alpha=0.3, ls=":")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT / "pca_factor_scores.png", dpi=140)
    print(f"\nSaved charts to {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="CSV/Excel of yields in percent; omit to use sample data")
    source.add_argument("--from-db", action="store_true", help="Load yields from yields.db (see collect.py)")
    parser.add_argument("--start", help="Earliest date to include when using --from-db (YYYY-MM-DD)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.from_db:
        from store import get_connection, load_yields as load_yields_db

        conn = get_connection()
        yields = load_yields_db(conn, start=args.start)
        conn.close()
        if yields.empty:
            raise SystemExit("No data in yields.db. Run collect.py first.")
    elif args.input:
        yields = load_yields(args.input)
    else:
        print("No --input given: using synthetic sample data.\n")
        yields = sample_yields()

    res = run_pca(yields)
    print_report(res)

    OUT.mkdir(exist_ok=True)
    res.loadings.to_csv(OUT / "loadings.csv")
    res.scores.to_csv(OUT / "factor_scores.csv")

    if not args.no_plots:
        make_plots(res)


if __name__ == "__main__":
    main()