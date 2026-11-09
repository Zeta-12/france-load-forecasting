import pandas as pd

from load_forecast import backtest, data, models
from tests.conftest import synthetic


def test_monthly_refit_never_sees_the_test_month(cfg, monkeypatch):
    load_df, wx = synthetic("2022-11-01", "2023-04-30")
    monkeypatch.setattr(data, "load", lambda cfg, src: (load_df, wx, wx))
    seen = []

    class Spy(models.NaiveWeek):
        def fit(self, X, y):
            seen.append(X["day"].max())
            return self

    monkeypatch.setattr(models, "make", lambda name, cfg: Spy())
    cfg = {**cfg, "history_start": "2022-11-15", "train": {"exclude": []}}
    cfg["backtest"] = {**cfg["backtest"], "start": "2023-02-01", "end": "2023-04-30"}
    pred = backtest.run(cfg, ["naive_week"], "day1")

    assert seen == [pd.Timestamp("2023-01-30"), pd.Timestamp("2023-02-27"), pd.Timestamp("2023-03-30")]
    assert pred["day"].min() == pd.Timestamp("2023-02-01")
    assert pred["day"].max() == pd.Timestamp("2023-04-30")
    assert pred["naive_week"].notna().all()


def test_excluded_period_is_not_used_for_fitting(cfg):
    load_df, wx = synthetic("2022-11-01", "2023-03-31")
    from load_forecast import features

    X = features.build(load_df["load"], wx, wx, cfg)
    cfg = {**cfg, "history_start": "2022-11-15", "train": {"exclude": [["2023-01-01", "2023-01-31"]]}}
    m = backtest.train_mask(X, load_df["load"], cfg, pd.Timestamp("2023-03-01"))
    days = X.loc[m, "day"]
    assert not days.between(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-31")).any()
    assert days.min() == pd.Timestamp("2022-11-15")
    assert days.max() == pd.Timestamp("2023-03-01")


def test_lightgbm_beats_naive_on_synthetic_data(cfg, monkeypatch):
    load_df, wx = synthetic("2022-06-01", "2023-03-31")
    monkeypatch.setattr(data, "load", lambda cfg, src: (load_df, wx, wx))
    cfg = {**cfg, "history_start": "2022-06-15", "train": {"exclude": []}}
    cfg["backtest"] = {**cfg["backtest"], "start": "2023-03-01", "end": "2023-03-31"}
    cfg["models"] = {**cfg["models"], "lightgbm": {**cfg["models"]["lightgbm"], "n_estimators": 200}}
    pred = backtest.run(cfg, ["naive_week", "lightgbm"], "day1")
    err = (pred[["naive_week", "lightgbm"]].sub(pred["y"], axis=0)).abs().mean()
    assert err["lightgbm"] < 0.5 * err["naive_week"]
