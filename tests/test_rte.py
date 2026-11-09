import numpy as np
import pandas as pd

from load_forecast import rte


def _raw(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["date_heure", "consommation", "prevision_j1", "prevision_j"])
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    path = tmp_path / "raw.parquet"
    df.to_parquet(path)
    return path


def test_duplicated_timestamps_are_averaged(tmp_path):
    path = _raw(
        tmp_path,
        [
            ("2024-03-31T01:00:00Z", 100.0, 110, 105),
            ("2024-03-31T01:00:00Z", 200.0, 210, 205),
            ("2024-03-31T01:30:00Z", 300.0, 310, 305),
        ],
    )
    df = rte.read_raw(path)
    assert len(df) == 2
    assert df.loc["2024-03-31 01:00", "load"].item() == 150.0


def test_hourly_uses_00_and_30_only(tmp_path):
    rows = []
    for m, v in [(0, 100), (15, 9999), (30, 200), (45, 9999)]:
        rows.append((f"2024-01-01T10:{m:02d}:00Z", v, v, v))
    df = rte.to_hourly(rte.read_raw(_raw(tmp_path, rows)))
    assert df["load"].iloc[0] == 150
    assert df["rte_j1"].iloc[0] == 150


def test_short_gaps_are_interpolated():
    idx = pd.date_range("2024-10-26 22:00", periods=8, freq="30min", tz="UTC")
    df = pd.DataFrame({"load": np.arange(8.0), "rte_j1": 1.0, "rte_j": 1.0}, index=idx).drop(idx[[4, 5]])
    hourly = rte.to_hourly(df)
    assert hourly["load"].notna().all()
    assert len(hourly) == 4
