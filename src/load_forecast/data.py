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
# weather is downloaded and saved one year at a time, so that an interrupted download resumes where it stopped
CHUNK_DAYS = 365


def _age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400 if path.exists() else float("inf")


def _start(store: Path, default: pd.Timestamp) -> pd.Timestamp:
    if not store.exists():
        return default
    last = pd.read_parquet(store, columns=["time"])["time"].max()
    return max(default, pd.Timestamp(last).tz_localize(None).normalize() - pd.Timedelta(days=3))


def chunks(start: pd.Timestamp, end: pd.Timestamp, days: int = CHUNK_DAYS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    while start <= end:
        stop = min(start + pd.Timedelta(days=days - 1), end)
        out.append((start, stop))
        start = stop + pd.Timedelta(days=1)
    return out


def _fetch_store(store: Path, start: pd.Timestamp, end: pd.Timestamp, fetch_chunk) -> None:
    for a, b in chunks(start, end):
        log.info("%s: %s to %s", store.stem, f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}")
        df = fetch_chunk(f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}", weather.pause_for((b - a).days + 1))
        weather.merge_store(store, df)


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
        model = wcfg["reanalysis_model"]
        _fetch_store(store, start, today, lambda a, b, pause: weather.fetch_archive(a, b, variables, model, pause))
    for name, days in FORECAST_SOURCES.items():
        if name not in sources:
            continue
        store = wdir / f"{name}.parquet"
        start = _start(store, pd.Timestamp(wcfg["forecast_start"]) - pd.Timedelta(days=10))

        def fetch_chunk(a, b, pause, days=days):
            return weather.fetch_previous_runs(a, b, variables, wcfg["forecast_model"], days, pause)

        _fetch_store(store, start, today, fetch_chunk)


def load(cfg: dict, forecast_source: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Hourly load (with RTE forecasts), observed national weather and, optionally, forecast national weather."""
    data_dir = Path(cfg["data_dir"])
    load_df = rte.load_hourly(data_dir)
    obs = weather.load_national(data_dir, "era5")
    fc = weather.load_national(data_dir, forecast_source) if forecast_source else None
    return load_df, obs, fc
