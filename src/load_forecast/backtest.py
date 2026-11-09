"""Rolling-origin backtest: refit at the start of every month on all the data known at that point."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from load_forecast import data, features, models, weather

log = logging.getLogger(__name__)


def tables(cfg: dict, weather_source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load table, features with observed weather (fit) and features with the evaluation weather (test)."""
    fc_source = None if weather_source == "obs" else weather_source
    load_df, obs, fc = data.load(cfg, fc_source)
    if fc is not None and cfg["weather"]["debias"]:
        fc = weather.debias(fc, obs, window_days=cfg["weather"]["debias_window_days"])
    load_df = features.regular(load_df, pd.Timestamp(cfg["history_start"]) - pd.Timedelta(days=14))
    X_fit = features.build(load_df["load"], obs, obs, cfg)
    X_test = X_fit if fc is None else features.build(load_df["load"], obs, fc, cfg)
    return load_df, X_fit, X_test


def train_mask(X: pd.DataFrame, y: pd.Series, cfg: dict, last_day: pd.Timestamp) -> np.ndarray:
    m = (X["day"] >= pd.Timestamp(cfg["history_start"])) & (X["day"] <= last_day) & y.notna()
    for a, b in cfg["train"]["exclude"]:
        m &= ~X["day"].between(pd.Timestamp(a), pd.Timestamp(b))
    return m.to_numpy()


def fit(name: str, cfg: dict, X_fit: pd.DataFrame, y: pd.Series, last_day: pd.Timestamp):
    """Model trained on every day up to `last_day` (the last complete day at issue time)."""
    m = train_mask(X_fit, y, cfg, last_day)
    t0 = time.perf_counter()
    model = models.make(name, cfg).fit(X_fit[m], y[m].to_numpy())
    log.info("%s fitted up to %s: %d rows, %.1f s", name, f"{last_day:%Y-%m-%d}", m.sum(), time.perf_counter() - t0)
    return model


def run_tables(
    load_df: pd.DataFrame,
    X_fit: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: dict,
    model_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    y = load_df["load"]
    out = []
    for month in pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS"):
        test = (X_test["day"] >= max(month, start)) & (X_test["day"] < month + pd.offsets.MonthBegin(1))
        test &= (X_test["day"] <= end) & y.notna()
        if not test.any():
            continue
        res = pd.DataFrame(
            {
                "day": X_test.loc[test, "day"],
                "hour": X_test.loc[test, "hour"],
                "y": y[test],
                "rte_j1": load_df.loc[test, "rte_j1"],
            }
        )
        for name in model_names:
            # day D is forecast on D-1, when the last complete day is D-2
            model = fit(name, cfg, X_fit, y, month - pd.Timedelta(days=2))
            res[name] = model.predict(X_test[test])
        out.append(res)
    return pd.concat(out)


def run(cfg: dict, model_names: list[str] | None = None, weather_source: str | None = None) -> pd.DataFrame:
    bt = cfg["backtest"]
    load_df, X_fit, X_test = tables(cfg, weather_source or bt["weather"])
    start, end = pd.Timestamp(bt["start"]), pd.Timestamp(bt["end"])
    return run_tables(load_df, X_fit, X_test, cfg, model_names or bt["models"], start, end)


def save(pred: pd.DataFrame, cfg: dict, tag: str) -> Path:
    path = Path(cfg["output_dir"]) / f"backtest_{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(path)
    return path


def read(cfg: dict, tag: str) -> pd.DataFrame:
    return pd.read_parquet(Path(cfg["output_dir"]) / f"backtest_{tag}.parquet")
