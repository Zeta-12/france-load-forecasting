"""Tables and figures of the README, from the saved backtests."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_forecast import conformal, features, metrics, rte
from load_forecast.calendar import public_holidays

NAMES = {
    "naive_week": "Same hour last week",
    "lightgbm": "LightGBM",
    "lightgbm_abs": "LightGBM, absolute load target",
    "catboost": "CatBoost",
    "mean_gbm": "Mean of LightGBM and CatBoost",
    "rte_j1": "RTE J-1, as published",
    "rte_adj": "RTE J-1, level-adjusted",
    "blend_rte": "Mean of this model and RTE J-1 (adjusted)",
    "blend_raw": "Mean of this model and RTE J-1",
}
COLORS = {
    "lightgbm": "#2a78d6",
    "mean_gbm": "#2a78d6",
    "rte_adj": "#eb6834",
    "rte_j1": "#eb6834",
    "catboost": "#1baf7a",
    "actual": "#0b0b0b",
    "band": "#cde2fb",
}
MUTED = "#52514e"
# offset window for RTE's J-1, see conformal.rolling_offset
RTE_WINDOW = 28
ACI = "Adaptive conformal (ACI), used live"


def _style(ax):
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#b8b7b2")
    ax.spines["bottom"].set_color("#b8b7b2")
    ax.tick_params(colors=MUTED)


def rte_adjusted(cfg: dict, index: pd.DatetimeIndex) -> pd.Series:
    """RTE's J-1 shifted by its mean gap to the consolidated load over the last 28 days (up to D-2), per hour."""
    h = rte.load_hourly(Path(cfg["data_dir"]))
    h = h[(h.index >= index.min() - pd.Timedelta(days=RTE_WINDOW + 5)) & (h.index <= index.max())]
    day, hour = features.local_day(h.index, cfg["timezone"])
    df = pd.DataFrame({"day": day, "hour": hour, "y": h["load"], "rte_j1": h["rte_j1"]}, index=h.index)
    return conformal.rolling_offset(df, "rte_j1", RTE_WINDOW).reindex(index)


def load(cfg: dict, tag: str) -> pd.DataFrame:
    out = Path(cfg["output_dir"])
    pred = pd.read_parquet(out / f"backtest_{tag}.parquet")
    # models run separately on the same rows (backtest --tag <tag>_<name>)
    for extra in sorted(out.glob(f"backtest_{tag}_cat.parquet")):
        other = pd.read_parquet(extra)
        for col in other.columns.difference(pred.columns):
            pred[col] = other[col].reindex(pred.index)
    pred["rte_adj"] = rte_adjusted(cfg, pred.index)
    if {"lightgbm", "catboost"} <= set(pred.columns):
        pred["mean_gbm"] = (pred["lightgbm"] + pred["catboost"]) / 2
    if "lightgbm" in pred.columns:
        pred["blend_rte"] = (pred[main(pred)] + pred["rte_adj"]) / 2
    return pred


def main(pred: pd.DataFrame) -> str:
    """The forecast used in production: mean of LightGBM and CatBoost when both were run."""
    return "mean_gbm" if "mean_gbm" in pred else "lightgbm"


def md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join([""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for c in cols:
            v = row[c]
            cells.append(floatfmt.get(c, "{:.0f}").format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def point_table(pred: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    pred = pred.dropna(subset=cols)
    t = metrics.table(pred, cols)
    t["peak_mae"] = [metrics.daily_peak_error(pred, c).abs().mean() for c in cols]
    t.index = [NAMES.get(c, c) for c in cols]
    t = t[["mape", "mae", "rmse", "bias", "peak_mae"]]
    t.columns = ["MAPE (%)", "MAE (MW)", "RMSE (MW)", "Bias (MW)", "Daily peak MAE (MW)"]
    return t


def worst_days(pred: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    hol = public_holidays(range(pred["day"].min().year, pred["day"].max().year + 1))
    g = pred.groupby("day")
    out = pd.DataFrame(
        {
            "MAPE (%)": g.apply(lambda d: metrics.point_metrics(d["y"], d[col])["mape"]),
            "RTE J-1 MAPE (%)": g.apply(lambda d: metrics.point_metrics(d["y"], d["rte_j1"])["mape"]),
            "Bias (MW)": g.apply(lambda d: (d[col] - d["y"]).mean()),
        }
    )
    out = out.sort_values("MAPE (%)", ascending=False).head(n)
    out["Day"] = [f"{d:%a}" + (" (holiday)" if d.date() in hol else "") for d in out.index]
    out.index = [f"{d:%Y-%m-%d}" for d in out.index]
    return out[["Day", "MAPE (%)", "RTE J-1 MAPE (%)", "Bias (MW)"]]


def interval_methods(pred: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    c = cfg["conformal"]
    alpha, window, gamma = c["alpha"], c["window_days"], c["gamma"]
    col = main(pred)
    out = {}
    if {"lightgbm_q05", "lightgbm_q95"} <= set(pred.columns):
        out["LightGBM quantile regression"] = pred[["lightgbm_q05", "lightgbm_q95"]].set_axis(["lo", "hi"], axis=1)
    out["Rolling conformal, per hour"] = conformal.rolling(pred, col, alpha, window, by_hour=True)[0]
    out["Rolling conformal"] = conformal.rolling(pred, col, alpha, window)[0]
    out[ACI] = conformal.rolling(pred, col, alpha, window, by_hour=c["by_hour"], gamma=gamma)[0]
    if {"lightgbm_q05", "lightgbm_q95"} <= set(pred.columns):
        out["Conformalized quantile regression"] = conformal.cqr(pred, "lightgbm_q05", "lightgbm_q95", alpha, window)
    return out


def interval_table(pred: pd.DataFrame, methods: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    # same rows for every method: those where all intervals exist
    ok = np.logical_and.reduce([m["lo"].notna().to_numpy() for m in methods.values()])
    rows = {}
    for name, iv in methods.items():
        m = metrics.interval_metrics(pred["y"][ok], iv["lo"][ok], iv["hi"][ok], alpha)
        monthly = pred[ok].assign(inside=((pred["y"] >= iv["lo"]) & (pred["y"] <= iv["hi"]))[ok])
        cov_m = monthly.groupby(monthly["day"].dt.to_period("M"))["inside"].mean()
        rows[name] = {
            "Coverage (%)": m["coverage"] * 100,
            "Worst month (%)": cov_m.min() * 100,
            "Mean width (MW)": m["width"],
            "Width (% of load)": m["width_pct"],
            "Interval score (MW)": m["interval_score"],
        }
    return pd.DataFrame(rows).T


def fig_week(pred: pd.DataFrame, iv: pd.DataFrame | None, tz: str, path: Path, start=None):
    """One week: actual, the model with its interval, RTE J-1. Default: the week with the highest load."""
    if start is None:
        daily = pred.groupby("day")["y"].mean()
        start = daily.rolling(7).mean().idxmax() - pd.Timedelta(days=6)
    start = pd.Timestamp(start)
    w = pred[(pred["day"] >= start) & (pred["day"] < start + pd.Timedelta(days=7))]
    t = w.index.tz_convert(tz).tz_localize(None)
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    if iv is not None:
        lo, hi = iv.loc[w.index, "lo"] / 1e3, iv.loc[w.index, "hi"] / 1e3
        ax.fill_between(t, lo, hi, color=COLORS["band"], lw=0, label="90 % interval")
    ax.plot(t, w["y"] / 1e3, color=COLORS["actual"], lw=1.6, label="Actual")
    ax.plot(t, w[main(w)] / 1e3, color=COLORS["lightgbm"], lw=1.6, label="This model")
    ax.plot(t, w["rte_adj"] / 1e3, color=COLORS["rte_j1"], lw=1.4, ls=(0, (2, 1.5)), label="RTE J-1, level-adjusted")
    ax.set_ylabel("GW")
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.15))
    _style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return start


def fig_grouped(table: pd.DataFrame, path: Path, xlabel: str, ylabel: str = "MAPE (%)"):
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    monthly = isinstance(table.index, pd.PeriodIndex)
    x = table.index.to_timestamp() if monthly else np.arange(len(table))
    for col in table.columns:
        color = COLORS.get(col, MUTED)
        ax.plot(x, table[col], color=color, lw=2, marker="o", ms=4, label=NAMES.get(col, col))
    if not monthly:
        ax.set_xticks(x, [str(i) for i in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=len(table.columns), loc="upper left", bbox_to_anchor=(0, 1.15))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_coverage(pred: pd.DataFrame, methods: dict[str, pd.DataFrame], alpha: float, path: Path, names: list[str]):
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    colors = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a"]
    for name, color in zip(names, colors, strict=False):
        iv = methods[name]
        inside = ((pred["y"] >= iv["lo"]) & (pred["y"] <= iv["hi"])).where(iv["lo"].notna())
        m = inside.groupby(pred["day"].dt.to_period("M")).mean().dropna() * 100
        ax.plot(m.index.to_timestamp(), m.to_numpy(), color=color, lw=2, marker="o", ms=4, label=name)
    ax.axhline((1 - alpha) * 100, color=MUTED, lw=1, ls="--")
    ax.set_ylabel("Monthly coverage (%)")
    ax.legend(frameon=False, ncol=len(names), loc="upper left", bbox_to_anchor=(0, 1.15))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _maybe(cfg: dict, tag: str) -> pd.DataFrame | None:
    path = Path(cfg["output_dir"]) / f"backtest_{tag}.parquet"
    return load(cfg, tag) if path.exists() else None


def write_all(cfg: dict, out: Path) -> str:
    """Every table and figure of the README. Returns the tables as markdown."""
    out.mkdir(parents=True, exist_ok=True)
    tz, alpha = cfg["timezone"], cfg["conformal"]["alpha"]
    pred = load(cfg, "day1")
    fmt = {"MAPE (%)": "{:.2f}", "RTE J-1 MAPE (%)": "{:.2f}"}
    parts = []

    cols = [
        c for c in ["naive_week", "lightgbm", "catboost", "mean_gbm", "rte_j1", "rte_adj", "blend_rte"] if c in pred
    ]
    first, last = pred["day"].min(), pred["day"].max()
    parts.append(
        f"### Point forecasts, {first:%Y-%m-%d} to {last:%Y-%m-%d}\n\n" + md_table(point_table(pred, cols), fmt)
    )

    rows = {}
    for tag, label in [
        ("obs", "ERA5 (observed weather, upper bound)"),
        ("day1", "ECMWF IFS day-1 forecast, debiased"),
        ("day1_raw", "ECMWF IFS day-1 forecast, raw"),
        ("day2", "ECMWF IFS day-2 forecast, debiased"),
    ]:
        p = pred if tag == "day1" else _maybe(cfg, tag)
        if p is not None:
            rows[label] = point_table(p.loc[p.index.intersection(pred.index)], ["lightgbm"]).iloc[0]
    if len(rows) > 1:
        parts.append("### Weather input of the target day (LightGBM)\n\n" + md_table(pd.DataFrame(rows).T, fmt))

    rt = _maybe(cfg, "realtime")
    if rt is not None:
        # on the real-time series RTE's J-1 needs no adjustment
        rt["blend_raw"] = (rt[main(rt)] + rt["rte_j1"]) / 2
        rt_cols = [c for c in ["lightgbm_abs", "lightgbm", "catboost", "mean_gbm", "rte_j1", "blend_raw"] if c in rt]
        span = f"{rt['day'].min():%Y-%m-%d} to {rt['day'].max():%Y-%m-%d}"
        parts.append(f"### Real-time series, {span}\n\n" + md_table(point_table(rt, rt_cols), fmt))

    abl = _maybe(cfg, "day1_abs")
    if abl is not None and "lightgbm_abs" in abl:
        both = pred[["y", "day", "hour", "lightgbm"]].join(abl[["lightgbm_abs"]], how="inner")
        t = point_table(both, ["lightgbm_abs", "lightgbm"])
        t.index = ["Absolute load target", "Target relative to last week (default)"]
        parts.append("### Target (LightGBM, day-1 weather)\n\n" + md_table(t, fmt))

    col = main(pred)
    both = pred.dropna(subset=[col, "rte_adj"])
    by_month = metrics.grouped(both, [col, "rte_adj"], both["day"].dt.to_period("M"))
    fig_grouped(by_month, out / "mape_by_month.png", "")
    by_hour = metrics.grouped(both, [col, "rte_adj"], "hour")
    fig_grouped(by_hour, out / "mape_by_hour.png", "Hour of the day (Paris)")
    parts.append(f"### Worst days ({NAMES[col]})\n\n" + md_table(worst_days(pred, col), fmt))

    methods = interval_methods(pred, cfg)
    iv_table = interval_table(pred, methods, alpha)
    parts.append(
        f"### {int(round((1 - alpha) * 100))} % intervals\n\n"
        + md_table(iv_table, {"Coverage (%)": "{:.1f}", "Worst month (%)": "{:.1f}", "Width (% of load)": "{:.1f}"})
    )
    shown = [n for n in methods if n != "Rolling conformal, per hour" and "Conformalized" not in n]
    fig_coverage(pred, methods, alpha, out / "interval_coverage.png", shown)
    fig_week(pred, methods[ACI], tz, out / "week.png")
    fig_rte_gap(cfg, out / "rte_gap.png")

    text = "\n\n".join(parts) + "\n"
    (out / "results.md").write_text(text, encoding="utf-8")
    return text


def fig_rte_gap(cfg: dict, path: Path) -> pd.DataFrame:
    """Monthly mean of RTE J-1 - actual, against the consolidated series and against the real-time one."""
    data_dir = Path(cfg["data_dir"])
    rt = rte.load_realtime(data_dir).dropna(subset=["load"])
    hist = rte.load_hourly(data_dir)
    cons = hist[(hist.index >= pd.Timestamp(cfg["history_start"], tz="UTC")) & (hist.index < rt.index.min())]
    tz = cfg["timezone"]
    monthly = {
        "consolidated": (cons["rte_j1"] - cons["load"]).tz_convert(tz).resample("MS").mean() / 1e3,
        "real-time": (rt["rte_j1"] - rt["load"]).tz_convert(tz).resample("MS").mean() / 1e3,
    }
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    ax.axhline(0, color="#b8b7b2", lw=1)
    m = monthly["consolidated"]
    t = m.index.tz_localize(None)
    ax.plot(t, m.to_numpy(), color=COLORS["rte_j1"], lw=1.8, label="against consolidated data")
    m = monthly["real-time"]
    t = m.index.tz_localize(None)
    ax.plot(t, m.to_numpy(), "o", color=COLORS["lightgbm"], ms=6, label="against real-time data")
    ax.set_ylabel("RTE J-1 minus actual (GW)")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.17))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return pd.DataFrame(monthly)
