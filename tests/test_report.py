from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast import report, rte


def _backtest(cfg, n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-04-01", periods=n_days * 24, freq="1h", tz="UTC", name="time")
    local = idx.tz_convert("Europe/Paris")
    y = 50000 + 5000 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 800, len(idx))
    rte_j1 = y - 1200 + rng.normal(0, 700, len(idx))
    pred = pd.DataFrame(
        {"day": local.tz_localize(None).normalize(), "hour": local.hour, "y": y, "rte_j1": rte_j1}, index=idx
    )
    pred["naive_week"] = pd.Series(y, index=idx).shift(168).to_numpy()
    for name, sd in [("lightgbm", 700), ("catboost", 650)]:
        pred[name] = y + rng.normal(0, sd, len(idx))
    pred["lightgbm_q05"] = pred["lightgbm"] - 1000
    pred["lightgbm_q95"] = pred["lightgbm"] + 1000
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out / "backtest_day1.parquet")
    return pred


def test_write_all(cfg, tmp_path, monkeypatch):
    pred = _backtest(cfg)
    hourly = pred[["y", "rte_j1"]].rename(columns={"y": "load"})
    monkeypatch.setattr(rte, "load_hourly", lambda data_dir: hourly)
    monkeypatch.setattr(rte, "load_realtime", lambda data_dir: hourly[hourly.index >= "2024-07-01"])
    text = report.write_all(cfg, tmp_path / "assets")
    figures = ["week.png", "mape_by_month.png", "mape_by_hour.png", "interval_coverage.png", "rte_gap.png"]
    for name in [*figures, "results.md"]:
        assert (tmp_path / "assets" / name).exists()
    assert "RTE J-1, level-adjusted" in text
    assert "Adaptive conformal (ACI), used live" in text


def test_level_adjustment_removes_rte_offset(cfg, monkeypatch):
    pred = _backtest(cfg)
    monkeypatch.setattr(rte, "load_hourly", lambda data_dir: pred[["y", "rte_j1"]].rename(columns={"y": "load"}))
    loaded = report.load(cfg, "day1")
    t = report.point_table(loaded, ["rte_j1", "rte_adj"])
    assert t.loc["RTE J-1, as published", "Bias (MW)"] < -1000
    assert abs(t.loc["RTE J-1, level-adjusted", "Bias (MW)"]) < 150


def test_md_table():
    df = pd.DataFrame({"MAPE (%)": [1.234], "MAE (MW)": [812.4], "Day": ["Mon"]}, index=["a"])
    md = report.md_table(df, {"MAPE (%)": "{:.2f}"})
    assert md.splitlines()[2] == "| a | 1.23 | 812 | Mon |"
