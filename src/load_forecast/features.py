"""Feature table for day-ahead forecasting.

The forecast for local day D is issued on D-1 at 10:00. A row of day D only uses:
- consumption up to D-1 at `last_load_hour` (local time),
- observed weather before D,
- the weather of D itself from `target_wx`: observed when fitting, forecast when evaluating or in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_forecast.calendar import day_table

LOAD_FEATURES = ["lag_d1", "lag_d2", "lag_d7", "d1_last", "d1_morning", "d2_mean", "d7_mean"]
WEATHER_FEATURES = [
    "temp",
    "temp_ewm12",
    "temp_ewm48",
    "temp_day_mean",
    "temp_day_min",
    "temp_day_max",
    "temp_d2",
    "temp_d7",
    "solar",
    "wind",
]
# no cloud cover: IFS and ERA5 don't agree on it at all
CALENDAR_FEATURES = ["hour", "dow", "doy", "holiday", "holiday_prev", "holiday_next", "bridge", "xmas"]
# was D-2 / D-7 a holiday (the lags come from these days)
LAG_DAY_FEATURES = ["off_d2", "off_d7"]
FEATURES = CALENDAR_FEATURES + LAG_DAY_FEATURES + LOAD_FEATURES + WEATHER_FEATURES

EWM_HALFLIFE = {"temp_ewm12": 12, "temp_ewm48": 48}


def utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def hourly_index(start, end) -> pd.DatetimeIndex:
    return pd.date_range(utc(start), utc(end), freq="1h", name="time")


def local_day(index: pd.DatetimeIndex, tz: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    local = index.tz_convert(tz)
    return local.tz_localize(None).normalize(), local.hour.to_numpy()


def _by_day(values: pd.Series, day: pd.DatetimeIndex, how: str) -> pd.Series:
    return pd.Series(values.to_numpy(), index=day).groupby(level=0).agg(how)


def _shift_days(per_day: pd.Series, day: pd.DatetimeIndex, k: int) -> np.ndarray:
    return per_day.reindex(day - pd.Timedelta(days=k)).to_numpy()


def continued_ewm(obs: pd.Series, target: pd.Series, day: pd.DatetimeIndex, halflife: float) -> np.ndarray:
    """EWM of the observed series up to the end of D-1, then continued over D with the target-day values."""
    a = 1 - 0.5 ** (1 / halflife)
    b = 1 - a
    s_obs = obs.ewm(alpha=a, adjust=False, ignore_na=True).mean().to_numpy()
    day_s = pd.Series(day)
    k = day_s.groupby(day_s).cumcount().to_numpy()
    first = np.flatnonzero(k == 0)
    prev = np.where(first > 0, s_obs[np.maximum(first - 1, 0)], np.nan)
    start_state = pd.Series(prev, index=day[first]).reindex(day).to_numpy()
    x = target.to_numpy() * b ** (-k.astype(float))
    cum = pd.Series(x).groupby(day_s).cumsum().to_numpy()
    return b ** (k + 1) * start_state + a * b**k * cum


def build(load: pd.Series, obs: pd.DataFrame, target_wx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per hour of `load.index` (regular hourly, UTC)."""
    tz = cfg["timezone"]
    cut = cfg["last_load_hour"]
    index = load.index
    obs = obs.reindex(index)
    tgt = target_wx.reindex(index)
    day, hour = local_day(index, tz)

    f = pd.DataFrame(index=index)
    f["day"] = day
    f["hour"] = hour
    days = pd.DatetimeIndex(day.unique())
    cal = day_table(days.union(days - pd.Timedelta(days=7)))
    for col in CALENDAR_FEATURES[1:]:
        f[col] = cal[col].reindex(day).to_numpy()
    # holiday, bridge or christmas week
    off = (cal["holiday"] | cal["bridge"] | cal["xmas"]).astype(int)
    f["off_d2"] = _shift_days(off, day, 2)
    f["off_d7"] = _shift_days(off, day, 7)

    # load: same hour 2 and 7 days before, same hour yesterday only if it is before the cut-off
    f["lag_d2"] = load.shift(48).to_numpy()
    f["lag_d7"] = load.shift(168).to_numpy()
    lag24_day, lag24_hour = local_day(index - pd.Timedelta(hours=24), tz)
    known = (lag24_day == day - pd.Timedelta(days=1)) & (lag24_hour <= cut)
    f["lag_d1"] = np.where(known, load.shift(24).to_numpy(), np.nan)
    daily = _by_day(load, day, "mean")
    morning = _by_day(load.where(hour <= cut), day, "mean")
    last = _by_day(load.where(hour == cut), day, "mean")
    f["d1_last"] = _shift_days(last, day, 1)
    f["d1_morning"] = _shift_days(morning, day, 1)
    f["d2_mean"] = _shift_days(daily, day, 2)
    f["d7_mean"] = _shift_days(daily, day, 7)

    # weather of the target day
    for col in ("temp", "solar", "wind"):
        f[col] = tgt[col].to_numpy()
    for name, hl in EWM_HALFLIFE.items():
        f[name] = continued_ewm(obs["temp"], tgt["temp"], day, hl)
    for how in ("mean", "min", "max"):
        f[f"temp_day_{how}"] = _by_day(tgt["temp"], day, how).reindex(day).to_numpy()
    obs_daily = _by_day(obs["temp"], day, "mean")
    f["temp_d2"] = _shift_days(obs_daily, day, 2)
    f["temp_d7"] = _shift_days(obs_daily, day, 7)
    return f


def regular(load: pd.DataFrame, start, end=None) -> pd.DataFrame:
    """Reindex the hourly load table on a regular UTC grid."""
    end = end if end is not None else load.index.max()
    return load.reindex(hourly_index(start, end))
