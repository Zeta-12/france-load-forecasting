"""Download and update the raw data (RTE + Open-Meteo), and load the hourly national tables."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from load_forecast import rte, weather

log = logging.getLogger(__name__)

# archived day-1 / day-2 forecasts, see weather.fetch_previous_runs
FORECAST_SOURCES = {"day1": 1, "day2": 2}


def _age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400 if path.exists() else float("inf")


def _start(store: Path, default: pd.Timestamp) -> pd.Timestamp:
    if not store.exists():
        return default
    last = pd.read_parquet(store, columns=["time"])["time"].max()
    return max(default, pd.Timestamp(last).tz_localize(None).normalize() - pd.Timedelta(days=3))


def fetch(cfg: dict, sources: list[str] | None = None, today: pd.Timestamp | None = None) -> None:
    data_dir = Path(cfg["data_dir"])
    sources = sources or ["rte", "era5", *FORECAST_SOURCES]
    if today is None:
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    wcfg = cfg["weather"]
    variables = wcfg["variables"]

    if "rte" in sources:
        raw = data_dir / "raw" / "rte"
        # the definitive file only changes once a month
        if _age_days(raw / "definitive.parquet") > 7:
            rte.download("definitive", raw / "definitive.parquet")
        rte.download("realtime", raw / "realtime.parquet")

    wdir = data_dir / "raw" / "weather"
    if "era5" in sources:
        store = wdir / "era5.parquet"
        start = _start(store, pd.Timestamp(cfg["history_start"]) - pd.Timedelta(days=31))
        # long requests weigh a lot on the open-meteo quota (600 per minute)
        pause = 40 if (today - start).days > 365 else 0
        df = weather.fetch_archive(f"{start:%Y-%m-%d}", f"{today:%Y-%m-%d}", variables, wcfg["reanalysis_model"], pause)
        weather.merge_store(store, df)
    for name, days in FORECAST_SOURCES.items():
        if name not in sources:
            continue
        store = wdir / f"{name}.parquet"
        start = _start(store, pd.Timestamp(wcfg["forecast_start"]) - pd.Timedelta(days=10))
        pause = 10 if (today - start).days > 180 else 0
        df = weather.fetch_previous_runs(
            f"{start:%Y-%m-%d}", f"{today:%Y-%m-%d}", variables, wcfg["forecast_model"], days, pause
        )
        weather.merge_store(store, df)


def load(cfg: dict, forecast_source: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Hourly load (with RTE forecasts), observed national weather and, optionally, forecast national weather."""
    data_dir = Path(cfg["data_dir"])
    load_df = rte.load_hourly(data_dir)
    obs = weather.load_national(data_dir, "era5")
    fc = weather.load_national(data_dir, forecast_source) if forecast_source else None
    return load_df, obs, fc
