"""Daily production run: forecast tomorrow, keep an archive of the forecasts and score it.

Run on D-1 in the morning (GitHub Actions cron). Steps:
1. update RTE data and ERA5, get the latest ECMWF forecast;
2. replay the last 60 days as a small backtest (same monthly refits, archived day-1 weather) to get out-of-sample
   residuals for the conformal intervals (ACI, see conformal.py);
3. fit LightGBM and CatBoost on everything up to D-2, forecast D with their mean, save forecasts/<D>.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast import backtest, conformal, data, features, rte, weather

log = logging.getLogger(__name__)

MODELS = ["lightgbm", "catboost"]


def _now_local(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp.now(tz=cfg["timezone"])


def forecast(
    cfg: dict, archive: Path, fetch: bool = True, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast of tomorrow (also written to the archive) and the replay of the last weeks."""
    tz = cfg["timezone"]
    now = now or _now_local(cfg)
    target = now.tz_localize(None).normalize() + pd.Timedelta(days=1)
    data_dir = Path(cfg["data_dir"])
    wcfg = cfg["weather"]
    if fetch:
        data.fetch(cfg, ["rte", "era5", "day1"])
    live_wx = weather.national(weather.fetch_forecast(wcfg["variables"], wcfg["forecast_model"], past_days=10))

    era5 = weather.load_national(data_dir, "era5")
    day1 = weather.load_national(data_dir, "day1")
    # archived day-1 forecasts for the past, the latest run for the coming hours
    fc = live_wx.combine_first(day1)
    if wcfg["debias"]:
        fc = weather.debias(fc, era5, window_days=wcfg["debias_window_days"])
        day1 = weather.debias(day1, era5, window_days=wcfg["debias_window_days"])
    # ERA5 is ~5 days behind, the forecast API fills the last days (and the rest of today)
    obs = era5.combine_first(live_wx)
    end = pd.Timestamp(target + pd.Timedelta(days=1), tz=tz).tz_convert("UTC") - pd.Timedelta(hours=1)
    start = pd.Timestamp(cfg["history_start"]) - pd.Timedelta(days=14)
    load_df = features.regular(rte.load_hourly(data_dir), start, end)
    # keep the recent weeks on the real-time series, even after RTE publishes a new consolidated month
    frozen = update_actuals(cfg, archive)["load"].reindex(load_df.index)
    load_df["load"] = frozen.combine_first(load_df["load"])
    y = load_df["load"]
    # nothing after the cut-off may leak in, even if the data is already there
    cutoff = pd.Timestamp(target - pd.Timedelta(days=1), tz=tz).replace(hour=cfg["last_load_hour"])
    y = y.where(y.index <= cutoff.tz_convert("UTC"))
    load_df["load"] = y

    X_fit = features.build(y, obs, obs, cfg)
    c = cfg["conformal"]
    cal_start = target - pd.Timedelta(days=c["replay_days"])
    X_cal = features.build(y, obs, day1, cfg)
    cal = backtest.run_tables(load_df, X_fit, X_cal, cfg, MODELS, cal_start, target - pd.Timedelta(days=2))
    cal["forecast"] = cal[MODELS].mean(axis=1)

    X_live = features.build(y, obs, fc, cfg)
    rows = X_live["day"] == target
    out = pd.DataFrame(
        {"day": target, "hour": X_live.loc[rows, "hour"], "y": float("nan"), "rte_j1": load_df.loc[rows, "rte_j1"]}
    )
    for name in MODELS:
        out[name] = backtest.fit(name, cfg, X_fit, y, target - pd.Timedelta(days=2)).predict(X_live[rows])
    out["forecast"] = out[MODELS].mean(axis=1)

    both = pd.concat([cal, out])
    iv, _ = conformal.rolling(both, "forecast", c["alpha"], c["window_days"], by_hour=c["by_hour"], gamma=c["gamma"])
    iv = iv.loc[out.index]
    table = pd.DataFrame(
        {
            "time": out.index.strftime("%Y-%m-%dT%H:%MZ"),
            "local_hour": out["hour"].to_numpy(),
            "forecast": out["forecast"].round(0).to_numpy(),
            "lo": iv["lo"].round(0).to_numpy(),
            "hi": iv["hi"].round(0).to_numpy(),
            "rte_j1": out["rte_j1"].round(0).to_numpy(),
        }
    )
    path = archive / "forecasts" / f"{target:%Y-%m-%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    log.info("forecast for %s written to %s (issued %s)", f"{target:%Y-%m-%d}", path, f"{now:%Y-%m-%d %H:%M}")
    replay = pd.DataFrame(
        {"time": cal.index, "day": cal["day"].dt.strftime("%Y-%m-%d"), "actual": cal["y"].to_numpy()}
    ).assign(forecast=cal["forecast"].to_numpy(), rte_j1=cal["rte_j1"].to_numpy())
    return table, replay


def update_actuals(cfg: dict, archive: Path) -> pd.DataFrame:
    """Keep the real-time load and RTE's J-1, as first published, in archive/actuals.csv.

    RTE replaces them with consolidated data a few months later (~1 GW higher since 2023, see the README).
    """
    path = archive / "actuals.csv"
    new = rte.load_realtime(Path(cfg["data_dir"]))[["load", "rte_j1"]].dropna(subset=["load"])
    # the last hour may only have its :00 value
    new = new.iloc[:-1]
    if path.exists():
        old = pd.read_csv(path, index_col="time")
        old.index = pd.to_datetime(old.index, utc=True)
        new = old.combine_first(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.round(1).to_csv(path, index_label="time", date_format="%Y-%m-%dT%H:%MZ")
    return new


def scored_archive(cfg: dict, archive: Path) -> pd.DataFrame:
    """All archived forecasts joined with the realised load (NaN while not known yet)."""
    files = sorted((archive / "forecasts").glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f).assign(day=f.stem) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    actuals = update_actuals(cfg, archive)
    t = pd.DatetimeIndex(df["time"])
    df["actual"] = actuals["load"].reindex(t).to_numpy()
    # RTE's J-1 may not have been published yet when the forecast was made
    published = actuals["rte_j1"].reindex(t).to_numpy()
    df["rte_j1"] = np.where(np.isnan(published), df["rte_j1"].to_numpy(dtype=float), published)
    return df
