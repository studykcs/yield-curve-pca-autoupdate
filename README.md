# Yield Curve PCA

An automated pipeline that pulls US Treasury yields from FRED, decomposes daily
yield-curve changes into level, slope and curvature via PCA, flags anomalous
days against those factors, and publishes a self-contained HTML dashboard to
GitHub Pages once a day.

**Live dashboard:** https://studykcs.github.io/yield-curve-pca-autoupdate/

## Pipeline

```
collect.py    fetch DGS1/2/3/5/10/20 from FRED, upsert into yields.db (SQLite)
     |
pca.py        covariance PCA on daily yield changes (bp) -> level/slope/curvature
     |
anomaly.py    per-factor z-score check against each PC's own eigenvalue,
     |        days that trip the threshold are tagged with hand-compiled
     |        macro context (rate decisions, crises, shocks)
     |
dashboard.py  renders charts + tables into one offline HTML file
     |        (output/dashboard.html and docs/index.html for Pages)
```

`run_pipeline.ps1` runs the four steps in order, logs to `output/pipeline.log`,
and commits/pushes the regenerated `docs/index.html` so the published
dashboard always reflects the latest trading day. It is scheduled to run
daily via Windows Task Scheduler.

## Method

PCA runs on *changes* in yield rather than levels, since yields are highly
persistent and near non-stationary — the covariance of levels would be
dominated by the trend rather than the day-to-day co-movement that level,
slope and curvature are meant to capture. The first three components
conventionally explain well over 95% of daily variance; `pca.py` reports the
explained-variance ratio, per-tenor R² and a sign check that confirms PC1 is
level (same sign across tenors), PC2 is slope (sign flips short vs long) and
PC3 is curvature (belly opposite the ends).

Anomaly detection (`anomaly.py`) is a per-factor sigma check — a day's PC1,
PC2 or PC3 score divided by that component's own standard deviation
(`sqrt(eigenvalue)`) — not a joint statistic. Flagged days are cross-referenced
against a hand-compiled table of macro events for context (not derived from
the data itself).

## Quick start

```bash
git clone https://github.com/studykcs/yield-curve-pca-autoupdate.git
cd yield-curve-pca-autoupdate
pip install pandas numpy requests python-dotenv plotly matplotlib

cp .env.example .env   # add your FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

python collect.py                  # fetch yields into yields.db
python anomaly.py --from-db        # detect anomalies (writes output/alerts.json)
python dashboard.py                # render output/dashboard.html + docs/index.html
```

`pca.py` can also be run standalone, either against synthetic sample data or
your own CSV/Excel file:

```bash
python pca.py                      # synthetic sample data
python pca.py --input yields.csv   # date column + one column per tenor, in percent
python pca.py --from-db            # read from yields.db instead
```

Python 3.10+.

## Files

| File | Role |
|---|---|
| `fred_data.py` | FRED API client — fetches constant-maturity Treasury series |
| `collect.py` | Fetches yields and upserts into `yields.db` |
| `store.py` | SQLite schema and read/write helpers (long-format table, date+tenor primary key) |
| `pca.py` | Covariance PCA engine, plus a synthetic sample generator and matplotlib report |
| `anomaly.py` | Z-score anomaly detection on PCA factor scores, with macro-event context |
| `dashboard.py` | Renders the self-contained Plotly HTML dashboard |
| `run_pipeline.ps1` | Daily automation: run the pipeline, commit and push the updated dashboard |

## Notes

- `docs/index.html` embeds the Plotly.js library directly (~6 MB) rather than
  loading it from a CDN, so the published dashboard renders standalone on
  GitHub Pages with no external script dependency.
- Project 01 of a broader set of quant finance projects; see
  [studykcs/financial-engineering-projects](https://github.com/studykcs/financial-engineering-projects)
  for interest-rate and FX derivatives models (Hull-White, SABR, KI-TRF,
  quanto options).
