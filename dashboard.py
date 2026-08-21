"""Single-file HTML dashboard for the yield curve PCA.

Reads yields from SQLite (see collect.py), runs PCA (see testcode.py) and
writes one self-contained HTML report: charts, data and the Plotly.js
library are all embedded in the file, so it opens directly in a browser
with no server and no internet connection needed.

Usage
-----
    python dashboard.py                    # writes output/dashboard.html
    python dashboard.py --start 2025-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from store import get_connection, load_yields
from testcode import OUT, PCAResult, reconstruction_r2, run_pca

PC_NAMES = {"PC1": "Level", "PC2": "Slope", "PC3": "Curvature"}


def build_curve_fig(yields: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in yields.columns:
        fig.add_scatter(x=yields.index, y=yields[col], mode="lines", name=col)
    fig.update_layout(title="Yield levels", xaxis_title="Date", yaxis_title="Yield (%)", template="plotly_white")
    return fig


def build_variance_fig(res: PCAResult) -> go.Figure:
    labels = [f"PC{i + 1}" for i in range(len(res.explained))]
    fig = go.Figure()
    fig.add_bar(x=labels, y=res.explained * 100, name="Explained variance (%)")
    fig.add_scatter(x=labels, y=res.cumulative * 100, name="Cumulative (%)", mode="lines+markers")
    fig.update_layout(title="Explained variance", yaxis_title="%", template="plotly_white")
    return fig


def build_loadings_fig(res: PCAResult) -> go.Figure:
    fig = go.Figure()
    for pc, label in PC_NAMES.items():
        fig.add_scatter(x=res.loadings.index, y=res.loadings[pc], mode="lines+markers", name=f"{pc} ({label})")
    fig.update_layout(title="Loadings by tenor", xaxis_title="Tenor", yaxis_title="Loading", template="plotly_white")
    return fig


def build_scores_fig(res: PCAResult) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[f"{pc} ({label})" for pc, label in PC_NAMES.items()],
    )
    for i, pc in enumerate(PC_NAMES, start=1):
        fig.add_scatter(x=res.scores.index, y=res.scores[pc], mode="lines", name=pc, row=i, col=1)
    fig.update_layout(height=700, template="plotly_white", showlegend=False, title="Factor scores over time (bp/day)")
    return fig


def render_dashboard(yields: pd.DataFrame, res: PCAResult) -> str:
    figs = {
        "Yield levels": build_curve_fig(yields),
        "Explained variance": build_variance_fig(res),
        "Loadings by tenor": build_loadings_fig(res),
        "Factor scores over time": build_scores_fig(res),
    }
    sections = "".join(
        f'<section><h2>{title}</h2>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</section>'
        for title, fig in figs.items()
    )

    r2 = reconstruction_r2(res)
    table_rows = "".join(f"<tr><td>{t}</td><td>{r2[t] * 100:.2f}%</td></tr>" for t in r2.index)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Yield Curve PCA Dashboard</title>
<script>{get_plotlyjs()}</script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; color: #1a1a1a; }}
h1 {{ margin-bottom: 0.2rem; }}
.meta {{ color: #666; margin-bottom: 2rem; }}
section {{ margin-bottom: 2.5rem; }}
table {{ border-collapse: collapse; }}
td, th {{ padding: 0.4rem 0.8rem; border-bottom: 1px solid #ddd; text-align: right; }}
</style>
</head>
<body>
<h1>Yield Curve PCA Dashboard</h1>
<p class="meta">
  {yields.index[0].date()} .. {yields.index[-1].date()} | {len(yields)} observations |
  generated {pd.Timestamp.now():%Y-%m-%d %H:%M}
</p>
{sections}
<section>
<h2>Variance explained by PC1-3, per tenor</h2>
<table><tr><th>Tenor</th><th>R&sup2;</th></tr>{table_rows}</table>
</section>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Earliest date to include (YYYY-MM-DD)")
    parser.add_argument("--out", default=None, help="Output HTML path (default: output/dashboard.html)")
    args = parser.parse_args()

    conn = get_connection()
    yields = load_yields(conn, start=args.start)
    conn.close()
    if yields.empty:
        raise SystemExit("No data in yields.db. Run collect.py first.")

    res = run_pca(yields)
    html = render_dashboard(yields, res)

    out_path = Path(args.out) if args.out else OUT / "dashboard.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
