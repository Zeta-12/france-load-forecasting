"""Weather from Open-Meteo (CC BY 4.0): ERA5 reanalysis, archived day-ahead forecasts and live forecasts.

National weather is a population-weighted mean over one point per mainland region. Corsica is left out, it is not
on RTE's network.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# region, city, lat, lon, population in millions (INSEE 2021, rounded)
REGIONS = [
    ("IDF", "Paris", 48.86, 2.35, 12.3),
    ("ARA", "Lyon", 45.76, 4.84, 8.1),
    ("NAQ", "Bordeaux", 44.84, -0.58, 6.0),
    ("HDF", "Lille", 50.63, 3.06, 6.0),
    ("OCC", "Toulouse", 43.60, 1.44, 6.0),
    ("GES", "Strasbourg", 48.57, 7.75, 5.6),
    ("PAC", "Marseille", 43.30, 5.37, 5.1),
    ("PDL", "Nantes", 47.22, -1.55, 3.8),
    ("BRE", "Rennes", 48.11, -1.68, 3.4),
    ("NOR", "Rouen", 49.44, 1.10, 3.3),
    ("BFC", "Dijon", 47.32, 5.04, 2.8),
    ("CVL", "Orleans", 47.90, 1.91, 2.6),
]

SHORT = {"temperature_2m": "temp", "cloud_cover": "cloud", "shortwave_radiation": "solar", "wind_speed_10m": "wind"}
DEBIAS = ["temp", "solar", "wind"]


_session = requests.Session()


def pause_for(days: int) -> float:
    """Seconds to wait between two requests of `days` days.

    Open-Meteo counts one call per location, 14 days and 10 variables, with a limit of 600 calls per minute.
    """
    return max(1.0, days / 14) / 9


def _get(url: str, params: dict, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=(20, 120))
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 15 * (attempt + 1)
            log.warning("%s on %s, retrying in %d s", type(e).__name__, url, wait)
            time.sleep(wait)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = 65 * (attempt + 1) if r.status_code == 429 else 15
            log.warning("open-meteo %s: %s, waiting %d s", r.status_code, r.text[:200], wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"open-meteo request failed after {retries} attempts: {url}")


def _params(lat: float, lon: float, hourly: list[str], model: str, **extra) -> dict:
    return {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly), "models": model, "timezone": "UTC", **extra}


def _frame(js: dict, variables: list[str], suffix: str = "") -> pd.DataFrame:
    h = js["hourly"]
    df = pd.DataFrame({v: h[v + suffix] for v in variables}, dtype="float64")
    df.index = pd.to_datetime(h["time"]).tz_localize("UTC")
    df.index.name = "time"
    return df


def _per_region(fetch_one, pause: float) -> pd.DataFrame:
    frames = []
    for i, (code, city, lat, lon, _) in enumerate(REGIONS):
        if i and pause:
            time.sleep(pause)
        log.info("weather %s (%s)", code, city)
        df = fetch_one(lat, lon)
        df["region"] = code
        frames.append(df)
    return pd.concat(frames).reset_index()


def fetch_archive(start: str, end: str, variables: list[str], model: str = "era5", pause: float = 0) -> pd.DataFrame:
    def one(lat, lon):
        p = _params(lat, lon, variables, model, start_date=start, end_date=end)
        return _frame(_get(ARCHIVE_URL, p), variables)

    return _per_region(one, pause)


def fetch_previous_runs(
    start: str, end: str, variables: list[str], model: str, days: int, pause: float = 0
) -> pd.DataFrame:
    """Archived forecasts issued `days` days before the valid time (Open-Meteo previous runs API)."""
    suffix = f"_previous_day{days}"

    def one(lat, lon):
        p = _params(lat, lon, [v + suffix for v in variables], model, start_date=start, end_date=end)
        return _frame(_get(PREVIOUS_RUNS_URL, p), variables, suffix)

    return _per_region(one, pause)


def fetch_forecast(variables: list[str], model: str, past_days: int = 7, forecast_days: int = 3) -> pd.DataFrame:
    def one(lat, lon):
        p = _params(lat, lon, variables, model, past_days=past_days, forecast_days=forecast_days)
        return _frame(_get(FORECAST_URL, p), variables)

    return _per_region(one, pause=pause_for(past_days + forecast_days))


def merge_store(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Append to a stored long table, new rows win."""
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new]).drop_duplicates(["time", "region"], keep="last")
    new = new.dropna(subset=[c for c in new.columns if c not in ("time", "region")], how="all")
    new = new.sort_values(["region", "time"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def national(long: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted mean over regions, one row per hour (NaN where a region is missing)."""
    weights = pd.Series({code: pop for code, _, _, _, pop in REGIONS})
    variables = [c for c in long.columns if c not in ("time", "region")]
    out = {}
    for v in variables:
        wide = long.pivot(index="time", columns="region", values=v)[weights.index].dropna()
        out[SHORT.get(v, v)] = (wide * weights).sum(axis=1) / weights.sum()
    df = pd.DataFrame(out).sort_index()
    df.index = pd.DatetimeIndex(df.index, name="time")
    return df.astype(np.float64)


def debias(fc: pd.DataFrame, obs: pd.DataFrame, columns=DEBIAS, window_days: int = 28, lag_days: int = 7):
    """Remove the mean forecast - ERA5 difference, per UTC hour, over days D-lag-window+1 .. D-lag.

    ERA5 comes out with a ~5 day delay, lag_days=7 stays on what is known at issue time.
    """
    out = fc.copy()
    for col in columns:
        diff = fc[col] - obs[col].reindex(fc.index)
        parts = [
            s.rolling(f"{window_days}D", min_periods=window_days // 2).mean().shift(lag_days, freq="D")
            for _, s in diff.groupby(diff.index.hour)
        ]
        bias = pd.concat(parts).reindex(fc.index)
        out[col] = fc[col] - bias.fillna(0.0)
    return out


def load_national(data_dir: Path, source: str) -> pd.DataFrame:
    path = data_dir / "raw" / "weather" / f"{source}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `flf fetch` first")
    return national(pd.read_parquet(path))
