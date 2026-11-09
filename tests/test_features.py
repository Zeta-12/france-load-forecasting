import numpy as np
import pandas as pd

from load_forecast import features
from load_forecast.calendar import day_table


def test_continued_ewm_matches_pandas_when_target_is_observed(toy):
    _, wx = toy
    day, _ = features.local_day(wx.index, "Europe/Paris")
    got = features.continued_ewm(wx["temp"], wx["temp"], day, 12)
    want = wx["temp"].ewm(halflife=12, adjust=False).mean().to_numpy()
    # the first local day has no state before it
    first_day = day == day[0]
    np.testing.assert_allclose(got[~first_day], want[~first_day], rtol=1e-9)


def test_no_information_after_issue_time(toy, cfg):
    """Changing anything after D-1 09:00 (load) or in D (observed weather) must not change the features of D."""
    load_df, wx = toy
    base = features.build(load_df["load"], wx, wx.copy(), cfg)
    target_day = pd.Timestamp("2023-03-15")
    rows = base["day"] == target_day

    load2 = load_df["load"].copy()
    cutoff = pd.Timestamp("2023-03-14 09:00", tz="Europe/Paris").tz_convert("UTC")
    load2[load2.index > cutoff] += 10_000
    obs2 = wx.copy()
    in_d = features.local_day(wx.index, "Europe/Paris")[0] >= target_day
    obs2.loc[in_d, "temp"] += 15
    changed = features.build(load2, obs2, wx.copy(), cfg)

    pd.testing.assert_frame_equal(base[rows], changed[rows])
    # and the change is visible two days later
    later = base["day"] == target_day + pd.Timedelta(days=2)
    assert not base.loc[later, "lag_d2"].equals(changed.loc[later, "lag_d2"])


def test_lag_d1_only_before_cutoff(toy, cfg):
    load_df, wx = toy
    f = features.build(load_df["load"], wx, wx, cfg)
    day = f[f["day"] == pd.Timestamp("2023-02-10")]
    assert day.loc[day["hour"] <= 9, "lag_d1"].notna().all()
    assert day.loc[day["hour"] > 9, "lag_d1"].isna().all()


def test_target_weather_is_used_for_the_day(toy, cfg):
    load_df, wx = toy
    fc = wx.copy()
    fc["temp"] -= 5
    a = features.build(load_df["load"], wx, wx, cfg)
    b = features.build(load_df["load"], wx, fc, cfg)
    np.testing.assert_allclose(b["temp"], a["temp"] - 5)
    np.testing.assert_allclose(b["temp_day_mean"], a["temp_day_mean"] - 5)
    # lagged observed temperature does not move
    np.testing.assert_allclose(b["temp_d2"].dropna(), a["temp_d2"].dropna())


def test_calendar():
    days = pd.date_range("2025-05-26", "2025-06-02", freq="D")
    d = day_table(days)
    assert d.loc["2025-05-29", "holiday"] == 1  # ascension
    assert d.loc["2025-05-30", "bridge"] == 1
    assert d.loc["2025-05-28", "bridge"] == 0
    x = day_table(pd.date_range("2025-12-23", "2026-01-02", freq="D"))
    assert x["xmas"].tolist() == [0] + [1] * 9 + [0]
