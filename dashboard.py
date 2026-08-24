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

from anomaly import MACRO_CONTEXT, detect_anomalies
from store import get_connection, load_yields
from testcode import OUT, PCAResult, reconstruction_r2, run_pca

PC_NAMES = {"PC1": "Level", "PC2": "Slope", "PC3": "Curvature"}

# Chart "paper" is intentionally fixed-light (not theme-reactive): Plotly
# renders to a static SVG/canvas, so re-theming per viewer isn't practical.
# Charts sit inside a light card on either page theme instead - see .chart-card.
CHART_PAPER = "#FBFAF7"
CHART_INK = "#17212C"
COLORWAY = ["#2E5C82", "#B3432B", "#3F7A5E", "#8A6D3B", "#5B4B8A", "#3D7A8A"]

STYLE = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --ink: #17212C;
  --ink-soft: #4B5A66;
  --paper: #F1F3F4;
  --surface: #FFFFFF;
  --border: #D7DCDF;
  --accent: #2E5C82;
  --accent-soft: #E4EBF1;
  --negative: #B3432B;
  --negative-soft: #F5E4DF;
  --positive: #3F7A5E;
  --positive-soft: #E3EEE8;
  --font-display: "Source Serif 4", Georgia, "Nanum Myeongjo", serif;
  --font-body: "IBM Plex Sans", "Noto Sans KR", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", "Noto Sans Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #E8EAEC;
    --ink-soft: #A7B2BA;
    --paper: #12171C;
    --surface: #1A2128;
    --border: #2B3540;
    --accent: #6FA0C7;
    --accent-soft: #223244;
    --negative: #E08066;
    --negative-soft: #3A241F;
    --positive: #7FBBA0;
    --positive-soft: #1F3229;
  }
}
:root[data-theme="dark"] {
  --ink: #E8EAEC;
  --ink-soft: #A7B2BA;
  --paper: #12171C;
  --surface: #1A2128;
  --border: #2B3540;
  --accent: #6FA0C7;
  --accent-soft: #223244;
  --negative: #E08066;
  --negative-soft: #3A241F;
  --positive: #7FBBA0;
  --positive-soft: #1F3229;
}
* { box-sizing: border-box; }
body {
  font-family: var(--font-body);
  background: var(--paper);
  color: var(--ink);
  margin: 0;
  padding: 2.5rem 1.5rem 4rem;
}
.page { max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 2.25rem; }
h1 {
  font-family: var(--font-display);
  font-weight: 700;
  font-size: 2rem;
  margin: 0 0 0.3rem;
  text-wrap: balance;
}
h2 {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 1.3rem;
  margin: 0 0 1rem;
  color: var(--ink);
}
.eyebrow {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.5rem;
}
.meta { color: var(--ink-soft); font-size: 0.95rem; margin: 0; }
.stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.stat { background: var(--surface); padding: 1rem 1.2rem; display: flex; flex-direction: column; gap: 0.3rem; }
.stat .label { font-size: 0.78rem; color: var(--ink-soft); }
.stat .value { font-family: var(--font-mono); font-size: 1.4rem; font-variant-numeric: tabular-nums; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.chart-card .plot-wrap { background: %(paper)s; border-radius: 8px; padding: 0.5rem; overflow-x: auto; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1.25rem; }
table { border-collapse: collapse; width: 100%%; font-variant-numeric: tabular-nums; }
th, td { padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); text-align: right; font-size: 0.92rem; }
th:first-child, td:first-child { text-align: left; }
th:nth-child(5), td:nth-child(5) { text-align: left; }
th { font-family: var(--font-mono); font-weight: 500; font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-soft); }
td { font-family: var(--font-mono); }
tr:last-child td { border-bottom: none; }
.chip { display: inline-flex; align-items: center; padding: 0.15rem 0.55rem; border-radius: 999px; font-family: var(--font-mono); font-size: 0.82rem; font-weight: 500; }
.chip.neg { background: var(--negative-soft); color: var(--negative); }
.chip.pos { background: var(--positive-soft); color: var(--positive); }
footer { color: var(--ink-soft); font-size: 0.85rem; border-top: 1px solid var(--border); padding-top: 1.25rem; }
a { color: var(--accent); }
@media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } body { padding: 1.5rem 1rem 3rem; } }
</style>
""" % {"paper": CHART_PAPER}


def _theme_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PAPER,
        font=dict(family="IBM Plex Sans, sans-serif", color=CHART_INK, size=12),
        colorway=COLORWAY,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#E3E0D8", zerolinecolor="#E3E0D8")
    fig.update_yaxes(gridcolor="#E3E0D8", zerolinecolor="#E3E0D8")
    return fig


def build_curve_fig(yields: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in yields.columns:
        fig.add_scatter(x=yields.index, y=yields[col], mode="lines", name=col, line=dict(width=1.6))
    fig.update_layout(title="Yield levels", xaxis_title="Date", yaxis_title="Yield (%)")
    return _theme_fig(fig)


def build_variance_fig(res: PCAResult) -> go.Figure:
    labels = [f"PC{i + 1}" for i in range(len(res.explained))]
    fig = go.Figure()
    fig.add_bar(x=labels, y=res.explained * 100, name="Explained variance (%)", marker_color=COLORWAY[0])
    fig.add_scatter(x=labels, y=res.cumulative * 100, name="Cumulative (%)", mode="lines+markers", line=dict(color=CHART_INK))
    fig.update_layout(title="Explained variance", yaxis_title="%")
    return _theme_fig(fig)


def build_loadings_fig(res: PCAResult) -> go.Figure:
    fig = go.Figure()
    for pc, label in PC_NAMES.items():
        fig.add_scatter(x=res.loadings.index, y=res.loadings[pc], mode="lines+markers", name=f"{pc} ({label})")
    fig.update_layout(title="Loadings by tenor", xaxis_title="Tenor", yaxis_title="Loading")
    return _theme_fig(fig)


def build_scores_fig(res: PCAResult, anomalies: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[f"{pc} ({label})" for pc, label in PC_NAMES.items()],
    )
    for i, pc in enumerate(PC_NAMES, start=1):
        fig.add_scatter(x=res.scores.index, y=res.scores[pc], mode="lines", name=pc, line=dict(width=1.3, color=COLORWAY[0]), row=i, col=1)
        hits = anomalies[anomalies["component"] == pc] if not anomalies.empty else anomalies
        if not hits.empty:
            fig.add_scatter(
                x=pd.to_datetime(hits["date"]), y=hits["move_bp"], mode="markers",
                marker=dict(color="#B3432B", size=9, symbol="x"), name=f"{pc} anomaly",
                showlegend=False, row=i, col=1,
            )
    fig.update_layout(height=700, showlegend=False, title="Factor scores over time (bp/day) — x = anomaly")
    return _theme_fig(fig)


def build_content(yields: pd.DataFrame, res: PCAResult, threshold: float = 3.0) -> str:
    """Inner page content shared by the standalone file and the Artifact publish."""
    anomalies = detect_anomalies(res, threshold=threshold)
    as_of = yields.index[-1].date()

    figs = {
        "수익률 수준 (Yield levels)": build_curve_fig(yields),
        "설명 분산 (Explained variance)": build_variance_fig(res),
        "만기별 로딩 (Loadings by tenor)": build_loadings_fig(res),
    }
    chart_cards = "".join(
        f'<div class="chart-card"><h2>{title}</h2>'
        f'<p class="meta">기준일: {as_of} · 매일 자동 재계산됨 (전체 구간 {yields.index[0].date()}–{as_of})</p>'
        f'<div class="plot-wrap">{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div></div>'
        for title, fig in figs.items()
    )
    scores_card = (
        f'<div class="chart-card"><h2>팩터 스코어 시계열 (Factor scores)</h2><div class="plot-wrap">'
        f'{pio.to_html(build_scores_fig(res, anomalies), include_plotlyjs=False, full_html=False)}</div></div>'
    )

    r2 = reconstruction_r2(res)
    r2_rows = "".join(f"<tr><td>{t}</td><td>{r2[t] * 100:.2f}%</td></tr>" for t in r2.index)

    if anomalies.empty:
        anomaly_body = f"<p>{threshold}-sigma를 넘는 이상치가 없습니다.</p>"
    else:
        anomaly_rows = "".join(
            f'<tr><td>{r.date.strftime("%Y-%m-%d")}</td><td>{r.label}</td>'
            f'<td><span class="chip {"neg" if r.z_score < 0 else "pos"}">{r.z_score:+.2f}</span></td>'
            f'<td>{r.move_bp:+.2f} bp</td>'
            f'<td>{MACRO_CONTEXT.get(r.date.strftime("%Y-%m-%d"), "확인 필요")}</td></tr>'
            for r in anomalies.itertuples()
        )
        anomaly_body = (
            "<table><tr><th>날짜</th><th>팩터</th><th>Z-score</th><th>변동폭</th>"
            f"<th>매크로 이슈</th></tr>{anomaly_rows}</table>"
        )

    latest = res.explained[:3] * 100

    return f"""
{STYLE}
<div class="page">
  <div>
    <p class="eyebrow">Fixed Income · PCA</p>
    <h1>수익률 곡선 PCA 대시보드</h1>
    <p class="meta">
      {yields.index[0].date()} – {yields.index[-1].date()} · {len(yields)}개 관측치 ·
      생성 시각 {pd.Timestamp.now():%Y-%m-%d %H:%M}
    </p>
  </div>

  <div class="stat-strip">
    <div class="stat"><span class="label">PC1 Level</span><span class="value">{latest[0]:.1f}%</span></div>
    <div class="stat"><span class="label">PC2 Slope</span><span class="value">{latest[1]:.1f}%</span></div>
    <div class="stat"><span class="label">PC3 Curvature</span><span class="value">{latest[2]:.1f}%</span></div>
    <div class="stat"><span class="label">누적 설명력</span><span class="value">{res.cumulative[2] * 100:.1f}%</span></div>
    <div class="stat"><span class="label">이상치 (&gt;{threshold}σ)</span><span class="value">{len(anomalies)}건</span></div>
  </div>

  <div class="grid-2">
    {chart_cards}
  </div>
  {scores_card}

  <div class="card">
    <h2>이상치 탐지 (&gt;{threshold}-sigma)</h2>
    {anomaly_body}
  </div>

  <div class="card">
    <h2>만기별 PC1–3 설명 분산 (R&sup2;)</h2>
    <table><tr><th>Tenor</th><th>R&sup2;</th></tr>{r2_rows}</table>
  </div>

  <footer>
    PCA는 만기별 금리의 <em>일간 변화량</em>(수준이 아님)의 공분산 고유분해로 계산됩니다.
    PC1(Level)·PC2(Slope)·PC3(Curvature)이 통상 전체 변동성의 95% 이상을 설명합니다.
    이상치는 각 팩터 스코어가 자기 자신의 표준편차 대비 {threshold}배를 넘는 날을 표시합니다.
    데이터 출처: FRED(세인트루이스 연은).
  </footer>
</div>
"""


def render_dashboard(yields: pd.DataFrame, res: PCAResult, threshold: float = 3.0) -> str:
    """Full standalone HTML file - opens directly in a browser, works offline."""
    content = build_content(yields, res, threshold)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>수익률 곡선 PCA 대시보드</title>
<script>{get_plotlyjs()}</script>
</head>
<body>
{content}
</body>
</html>"""


DOCS_DIR = Path(__file__).parent / "docs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Earliest date to include (YYYY-MM-DD)")
    parser.add_argument("--out", default=None, help="Output HTML path (default: output/dashboard.html)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Anomaly z-score threshold (default 3.0)")
    parser.add_argument("--no-docs", action="store_true", help="Skip writing docs/index.html (GitHub Pages source)")
    args = parser.parse_args()

    conn = get_connection()
    yields = load_yields(conn, start=args.start)
    conn.close()
    if yields.empty:
        raise SystemExit("No data in yields.db. Run collect.py first.")

    res = run_pca(yields)
    html = render_dashboard(yields, res, threshold=args.threshold)

    out_path = Path(args.out) if args.out else OUT / "dashboard.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")

    if not args.no_docs:
        DOCS_DIR.mkdir(exist_ok=True)
        docs_path = DOCS_DIR / "index.html"
        docs_path.write_text(html, encoding="utf-8")
        print(f"Wrote {docs_path}")


if __name__ == "__main__":
    main()
