"""Prediction intervals calibrated on out-of-sample residuals of the backtest.

When the forecast for day D is issued (on D-1 morning), the last day with complete actuals is D-2. Every method here
only uses residuals of days <= D-2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# minimum number of calibration residuals, as a fraction of a full window
MIN_FILL = 0.8


def _quantiles(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    """Finite-sample corrected lower / upper quantiles of signed residuals."""
    n = len(scores)
    lo_level = np.floor((n + 1) * alpha / 2) / n
    hi_level = np.ceil((n + 1) * (1 - alpha / 2)) / n
    lo = np.quantile(scores, lo_level, method="lower") if lo_level > 0 else -np.inf
    hi = np.quantile(scores, hi_level, method="higher") if hi_level <= 1 else np.inf
    return lo, hi


def _calibration_rows(days: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Row range [a, b) of the residuals usable for each row's day: days D-1-window to D-2."""
    a = np.searchsorted(days, days - np.timedelta64(window + 1, "D"), side="left")
    b = np.searchsorted(days, days - np.timedelta64(1, "D"), side="left")
    return a, b


def _day_groups(pred: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    pred = pred.sort_index()
    days = pred["day"].to_numpy(dtype="datetime64[ns]")
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    return pred, days, np.split(np.arange(len(pred)), starts[1:])


def rolling(
    pred: pd.DataFrame, col: str, alpha: float, window: int, by_hour: bool = False, gamma: float = 0.0
) -> tuple[pd.DataFrame, pd.Series]:
    """Split conformal on a sliding window of past residuals (y - forecast), optionally adaptive.

    by_hour: one pair of quantiles per hour of the day instead of pooling the 24 hours.
    gamma > 0: adaptive conformal inference (Gibbs and Candes, 2021) with daily steps and a 2-day delay,
    alpha_D = alpha_{D-1} + gamma * (alpha - miss rate of day D-2).
    Returns the intervals and the level used each day.
    """
    pred, days, groups = _day_groups(pred)
    hours = pred["hour"].to_numpy()
    res = (pred["y"] - pred[col]).to_numpy()
    a, b = _calibration_rows(days, window)
    lo = np.full(len(pred), np.nan)
    hi = np.full(len(pred), np.nan)
    levels = np.full(len(groups), alpha)
    miss = np.full(len(groups), np.nan)
    level = alpha
    for k, rows in enumerate(groups):
        if gamma and k >= 2 and not np.isnan(miss[k - 2]):
            level = float(np.clip(level + gamma * (alpha - miss[k - 2]), 0.005, 0.5))
        levels[k] = level
        i, j = a[rows[0]], b[rows[0]]
        ok = ~np.isnan(res[i:j])
        if ok.sum() < MIN_FILL * window * 24:
            continue
        scores, score_hours = res[i:j][ok], hours[i:j][ok]
        if by_hour:
            for h in np.unique(hours[rows]):
                r = rows[hours[rows] == h]
                lo[r], hi[r] = _quantiles(scores[score_hours == h], level)
        else:
            lo[rows], hi[rows] = _quantiles(scores, level)
        known = ~np.isnan(res[rows])
        if known.any():
            r = rows[known]
            miss[k] = np.mean((res[r] < lo[r]) | (res[r] > hi[r]))
    p = pred[col].to_numpy()
    out = pd.DataFrame({"lo": p + lo, "hi": p + hi}, index=pred.index)
    return out, pd.Series(levels, index=pd.DatetimeIndex(days[[g[0] for g in groups]]))


def cqr(pred: pd.DataFrame, lo_col: str, hi_col: str, alpha: float, window: int) -> pd.DataFrame:
    """Conformalized quantile regression (Romano et al., 2019) on a sliding window.

    The quantile regression band is widened (or narrowed) by the conformal quantile of max(lo - y, y - hi).
    """
    pred, days, groups = _day_groups(pred)
    lo_q, hi_q, y = (pred[c].to_numpy() for c in (lo_col, hi_col, "y"))
    scores = np.maximum(lo_q - y, y - hi_q)
    a, b = _calibration_rows(days, window)
    margin = np.full(len(pred), np.nan)
    for rows in groups:
        sc = scores[a[rows[0]] : b[rows[0]]]
        sc = sc[~np.isnan(sc)]
        n = len(sc)
        if n < MIN_FILL * window * 24:
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        margin[rows] = np.quantile(sc, level, method="higher")
    return pd.DataFrame({"lo": lo_q - margin, "hi": hi_q + margin}, index=pred.index)


def rolling_offset(pred: pd.DataFrame, col: str, window: int) -> pd.Series:
    """`col` shifted by its mean error per hour of the day over days D-1-window .. D-2 (RTE's J-1 level)."""
    pred, days, groups = _day_groups(pred)
    hours = pred["hour"].to_numpy()
    res = (pred["y"] - pred[col]).to_numpy()
    a, b = _calibration_rows(days, window)
    offset = np.full(len(pred), np.nan)
    for rows in groups:
        i, j = a[rows[0]], b[rows[0]]
        if j - i < MIN_FILL * window * 24:
            continue
        sc, sh = res[i:j], hours[i:j]
        for h in np.unique(hours[rows]):
            offset[rows[hours[rows] == h]] = np.nanmean(sc[sh == h])
    return pd.Series(pred[col].to_numpy() + offset, index=pred.index)
