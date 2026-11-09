"""YAML configuration with defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "data_dir": "data",
    "output_dir": "runs",
    # more threads is slower on a 4-core runner and on hybrid laptop CPUs
    "n_jobs": 4,
    "timezone": "Europe/Paris",
    "history_start": "2015-01-01",
    # the forecast for day D is issued on D-1 at 10:00 (local time), the last
    # consumption value known at that point is the one of last_load_hour
    "last_load_hour": 9,
    "weather": {
        "reanalysis_model": "era5",
        "forecast_model": "ecmwf_ifs025",
        # first day of the archived IFS forecasts (radiation only from 2024-03-06)
        "forecast_start": "2024-02-05",
        "variables": ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"],
        # see weather.debias
        "debias": True,
        "debias_window_days": 28,
    },
    "train": {
        # first covid lockdown, nothing like it in the test period
        "exclude": [["2020-03-16", "2020-05-11"]],
    },
    "backtest": {
        "start": "2024-04-01",
        "end": "2026-08-31",
        "models": ["naive_week", "lightgbm", "catboost"],
        "weather": "day1",
    },
    "models": {
        # target relative to the same hour last week, see models.GBM
        "lightgbm": {
            "anchor": "lag_d7",
            "relative": True,
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        # quantile objective is slow, smaller model for this baseline
        "lightgbm_quantile": {"n_estimators": 600, "learning_rate": 0.08},
        "catboost": {
            "anchor": "lag_d7",
            "relative": True,
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        },
    },
    "conformal": {
        "alpha": 0.1,
        "window_days": 28,
        "by_hour": False,
        "gamma": 0.01,
        # days replayed by the live run (window + a month of ACI updates)
        "replay_days": 60,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = DEFAULTS
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
