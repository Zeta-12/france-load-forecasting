"""Static dashboard (GitHub Pages) built from the forecast archive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast.metrics import interval_metrics, point_metrics

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
REPO = "https://github.com/Zeta-12/france-load-forecasting"


def scores(df: pd.DataFrame, alpha: float) -> list[dict]:
    done = df.dropna(subset=["actual"])
    if done.empty:
        return []
    last_day = done["day"].max()
    rows = []
    for label, days in [("Last 7 days", 7), ("Last 30 days", 30), ("Since start", None)]:
        d = done
        if days:
            first = (pd.Timestamp(last_day) - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
            d = done[done["day"] >= first]
        ours = point_metrics(d["actual"], d["forecast"])
        rte = point_metrics(d["actual"], d["rte_j1"])
        iv = interval_metrics(d["actual"], d["lo"], d["hi"], alpha)
        rows.append(
            {
                "period": label,
                "days": int(d["day"].nunique()),
                "mape": round(ours["mape"], 2),
                "rte_mape": round(rte["mape"], 2),
                "mae": round(ours["mae"]),
                "rte_mae": round(rte["mae"]),
                "coverage": round(iv["coverage"] * 100, 1),
            }
        )
    return rows


def _series(df: pd.DataFrame, tz: str) -> dict:
    t = df["time"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M").tolist()

    def col(c):
        return [None if np.isnan(v) else float(v) for v in df[c].to_numpy(dtype=float)]

    return {"t": t, **{c: col(c) for c in ("forecast", "lo", "hi", "rte_j1", "actual")}}


def replay_scores(replay: pd.DataFrame) -> dict:
    d = replay.dropna(subset=["actual", "forecast", "rte_j1"])
    return {
        "days": int(d["day"].nunique()),
        "first": d["day"].min(),
        "last": d["day"].max(),
        "mape": round(point_metrics(d["actual"], d["forecast"])["mape"], 2),
        "rte_mape": round(point_metrics(d["actual"], d["rte_j1"])["mape"], 2),
    }


def build(df: pd.DataFrame, out_dir: Path, cfg: dict, issued: pd.Timestamp, replay: pd.DataFrame | None = None) -> Path:
    tz = cfg["timezone"]
    alpha = cfg["conformal"]["alpha"]
    latest = df["day"].max()
    nxt = df[df["day"] == latest]
    recent_days = sorted(df.loc[df["actual"].notna(), "day"].unique())[-21:]
    recent = df[df["day"].isin(recent_days)]
    payload = {
        "issued": f"{issued:%Y-%m-%d %H:%M} ({tz})",
        "target": latest,
        "level": int(round((1 - alpha) * 100)),
        "next": _series(nxt, tz),
        "recent": _series(recent, tz),
        "scores": scores(df, alpha),
        "replay": None,
    }
    if replay is not None and len(replay):
        r = replay.assign(lo=np.nan, hi=np.nan)
        payload["replay"] = {"series": _series(r, tz), **replay_scores(replay)}
    out_dir.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__PLOTLY__", PLOTLY).replace("__REPO__", REPO)
    html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    df.assign(time=df["time"].dt.strftime("%Y-%m-%dT%H:%MZ")).to_csv(out_dir / "forecasts.csv", index=False)
    return path


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>France day-ahead load forecast</title>
<script src="__PLOTLY__"></script>
<style>
  :root { --bg: #fcfcfb; --fg: #0b0b0b; --muted: #52514e; --line: #e6e5e1; --ours: #2a78d6; --rte: #eb6834;
          --band: rgba(42, 120, 214, 0.18); --actual: #0b0b0b; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #1a1a19; --fg: #f0efec; --muted: #c3c2b7; --line: #34332f; --ours: #3987e5; --rte: #d95926;
            --band: rgba(57, 135, 229, 0.22); --actual: #f0efec; }
  }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
  main { max-width: 1000px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 17px; margin: 32px 0 8px; }
  p, .muted { color: var(--muted); }
  .chart { width: 100%; height: 380px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  .wrap { overflow-x: auto; }
  a { color: var(--ours); }
</style>
</head>
<body>
<main>
  <h1>France day-ahead electricity consumption</h1>
  <div class="muted" id="issued"></div>
  <p>Hourly forecast of mainland France consumption for tomorrow, made every morning (mean of a LightGBM and a
  CatBoost model, conformal intervals), next to RTE's own day-ahead (J-1) forecast. Method and backtest:
  <a href="__REPO__">github repo</a>.</p>

  <h2 id="next-title">Tomorrow</h2>
  <div id="next" class="chart"></div>

  <h2>Last 21 days</h2>
  <div id="recent" class="chart"></div>

  <div id="replay-block" hidden>
  <h2>Last weeks, replayed</h2>
  <p class="muted" id="replay-note"></p>
  <div id="replay" class="chart"></div>
  </div>

  <h2>Scores</h2>
  <div class="wrap"><table id="scores"></table></div>
  <p class="muted">MAPE / MAE against the realised consumption from RTE's real-time data (revised later by RTE).
  Coverage is the share of hours inside the interval.</p>

  <p class="muted">Data: RTE eCO2mix (Licence Ouverte v2.0), weather forecasts and ERA5 from
  <a href="https://open-meteo.com/">Open-Meteo</a> (CC BY 4.0, ECMWF IFS). Not affiliated with RTE.</p>
</main>
<script>
const D = __DATA__;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
document.getElementById("issued").textContent = "Issued " + D.issued;
document.getElementById("next-title").textContent = "Forecast for " + D.target;

function traces(s, withActual) {
  const out = [
    { x: s.t, y: s.hi, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x: s.t, y: s.lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: css("--band"),
      name: D.level + " % interval", hoverinfo: "skip" },
    { x: s.t, y: s.forecast, mode: "lines", name: "This model", line: { color: css("--ours"), width: 2 } },
    { x: s.t, y: s.rte_j1, mode: "lines", name: "RTE J-1", line: { color: css("--rte"), width: 1.5, dash: "dot" } },
  ];
  if (withActual) {
    out.push({ x: s.t, y: s.actual, mode: "lines", name: "Actual", line: { color: css("--actual"), width: 1.5 } });
  }
  return out;
}

function layout() {
  return {
    margin: { l: 56, r: 8, t: 8, b: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: css("--fg") }, hovermode: "x unified",
    yaxis: { title: "MW", gridcolor: css("--line"), zeroline: false },
    xaxis: { gridcolor: css("--line") }, legend: { orientation: "h", y: -0.15 },
  };
}

Plotly.newPlot("next", traces(D.next, false), layout(), { responsive: true, displayModeBar: false });
Plotly.newPlot("recent", traces(D.recent, true), layout(), { responsive: true, displayModeBar: false });
if (D.replay) {
  const r = D.replay;
  document.getElementById("replay-block").hidden = false;
  document.getElementById("replay-note").textContent = `Same model refitted every month and fed the archived ` +
    `day-ahead weather forecasts, ${r.first} to ${r.last} (${r.days} days). These forecasts were not issued live. ` +
    `MAPE ${r.mape} %, RTE J-1 ${r.rte_mape} %.`;
  Plotly.newPlot("replay", traces(r.series, true).slice(2), layout(), { responsive: true, displayModeBar: false });
}

const head = "<tr><th>Period</th><th>Days</th><th>MAPE</th><th>RTE J-1 MAPE</th><th>MAE (MW)</th>" +
  "<th>RTE J-1 MAE</th><th>Coverage</th></tr>";
const body = D.scores.map((r) => `<tr><td>${r.period}</td><td>${r.days}</td><td>${r.mape} %</td>` +
  `<td>${r.rte_mape} %</td><td>${r.mae}</td><td>${r.rte_mae}</td><td>${r.coverage} %</td></tr>`).join("");
document.getElementById("scores").innerHTML = head + (body || "<tr><td colspan=7>No realised day yet</td></tr>");
</script>
</body>
</html>
"""
