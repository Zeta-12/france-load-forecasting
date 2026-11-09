"""Point and interval metrics. Loads are in MW."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    e = p[ok] - y[ok]
    return {
        "mape": float(np.mean(np.abs(e) / y[ok]) * 100),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "n": int(ok.sum()),
    }


def table(pred: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: point_metrics(pred["y"], pred[c]) for c in cols}).T


def grouped(pred: pd.DataFrame, cols: list[str], key, metric: str = "mape") -> pd.DataFrame:
    """One row per group, one column per forecaster."""
    g = pred.groupby(key)
    return pd.DataFrame({c: g.apply(lambda d, c=c: point_metrics(d["y"], d[c])[metric]) for c in cols})


def daily_peak_error(pred: pd.DataFrame, col: str) -> pd.Series:
    """Error on the daily maximum (forecast peak - actual peak), MW."""
    g = pred.groupby("day")
    return g[col].max() - g["y"].max()


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    ok = ~(np.isnan(y) | np.isnan(lo) | np.isnan(hi))
    y, lo, hi = y[ok], lo[ok], hi[ok]
    width = hi - lo
    # interval (winkler) score: width + 2/alpha * distance outside
    score = width + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "width": float(np.mean(width)),
        "width_pct": float(np.mean(width / y) * 100),
        "interval_score": float(np.mean(score)),
        "n": int(ok.sum()),
    }
