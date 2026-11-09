"""RTE eCO2mix national data (consumption and RTE's own forecasts), from the ODRE open data portal.

Two datasets:
- eco2mix-national-cons-def: consolidated / definitive data since 2012, 30 min consumption.
- eco2mix-national-tr: real-time data for the current and previous month, 15 min.

Licence Ouverte v2.0 (Etalab), source: RTE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

ODRE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
DATASETS = {"definitive": "eco2mix-national-cons-def", "realtime": "eco2mix-national-tr"}
FIELDS = "date_heure,consommation,prevision_j1,prevision_j"
COLUMNS = {"consommation": "load", "prevision_j1": "rte_j1", "prevision_j": "rte_j"}


def download(kind: str, dest: Path, timeout: int = 600) -> Path:
    url = f"{ODRE}/{DATASETS[kind]}/exports/parquet"
    log.info("downloading %s", url)
    r = requests.get(url, params={"select": FIELDS}, timeout=timeout)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["date_heure"], utc=True)
    df = df.rename(columns=COLUMNS)[["time", *COLUMNS.values()]]
    for col in COLUMNS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    # the definitive dataset has duplicated UTC timestamps on spring DST days
    # (the non-existent local 02:xx is mapped onto 01:xx UTC), average them
    return df.groupby("time").mean().sort_index()


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean of the :00 and :30 values.

    Consumption is 30 min in the definitive data and 15 min in the real-time data, RTE's forecasts are 15 min.
    Using the same two samples per hour for everything keeps the target and RTE's forecast comparable.
    """
    df = df[df.index.minute.isin([0, 30])]
    hourly = df.resample("1h").mean()
    # autumn DST days miss 00:00 and 00:30 UTC in the definitive data
    return hourly.interpolate(limit=2, limit_area="inside")


def load_realtime(data_dir: Path) -> pd.DataFrame:
    return to_hourly(read_raw(data_dir / "raw" / "rte" / "realtime.parquet"))


def load_hourly(data_dir: Path) -> pd.DataFrame:
    """Definitive/consolidated history, completed with real-time data after its end."""
    raw = data_dir / "raw" / "rte"
    parts = [
        read_raw(raw / f"{kind}.parquet") for kind in ("definitive", "realtime") if (raw / f"{kind}.parquet").exists()
    ]
    if not parts:
        raise FileNotFoundError(f"no eCO2mix data in {raw}, run `flf fetch` first")
    df = parts[0]
    if len(parts) == 2:
        last = df["load"].last_valid_index()
        rt = parts[1]
        df = pd.concat([df[df.index <= last], rt[rt.index > last]])
    return to_hourly(df)
