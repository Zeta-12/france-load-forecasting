"""Calendar features on local (Europe/Paris) days."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def public_holidays(years: range) -> set:
    # national holidays only (no Alsace-Moselle extras)
    return set(holidays.country_holidays("FR", years=list(years)).keys())


def day_table(days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per local day (naive dates at midnight)."""
    hol = public_holidays(range(days.min().year - 1, days.max().year + 2))
    d = pd.DataFrame(index=days)
    dates = days.date
    d["dow"] = days.day_of_week
    d["doy"] = days.day_of_year
    d["holiday"] = np.isin(dates, list(hol)).astype(int)
    prev = (days - pd.Timedelta(days=1)).date
    nxt = (days + pd.Timedelta(days=1)).date
    d["holiday_prev"] = np.isin(prev, list(hol)).astype(int)
    d["holiday_next"] = np.isin(nxt, list(hol)).astype(int)
    # "pont": monday before a tuesday holiday, friday after a thursday holiday (ascension every year)
    monday_before = (d["dow"] == 0) & (d["holiday_next"] == 1)
    friday_after = (d["dow"] == 4) & (d["holiday_prev"] == 1)
    d["bridge"] = (monday_before | friday_after).astype(int)
    md = days.month * 100 + days.day
    d["xmas"] = ((md >= 1224) | (md <= 101)).astype(int)
    return d
