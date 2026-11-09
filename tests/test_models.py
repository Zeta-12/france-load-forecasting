import numpy as np
import pandas as pd

from load_forecast import conformal, features, models
from tests.conftest import synthetic


def _xy(cfg, shift=0.0):
    load_df, wx = synthetic("2021-01-01", "2023-03-31")
    y = load_df["load"].copy()
    # level drop of `shift` MW from 2023
    y[y.index >= pd.Timestamp("2023-01-01", tz="UTC")] -= shift
    X = features.build(y, wx, wx, cfg)
    return X, y


def test_anchored_model_follows_a_level_drop(cfg):
    X, y = _xy(cfg, shift=4000)
    fit = (X["day"] >= "2021-01-15") & (X["day"] < "2023-02-01")
    test = X["day"] >= "2023-03-01"
    params = {**cfg["models"]["lightgbm"], "n_estimators": 200}
    variants = {"raw": {"anchor": None, "relative": False}, "relative": {"anchor": "lag_d7", "relative": True}}
    bias = {}
    for name, extra in variants.items():
        m = models.GBM("lightgbm", {**params, **extra}).fit(X[fit], y[fit].to_numpy())
        bias[name] = np.mean(m.predict(X[test]) - y[test].to_numpy())
    # one month of the new level in the fit: the raw model stays too high, the relative one doesn't
    assert bias["raw"] > 1000
    assert abs(bias["relative"]) < 0.3 * bias["raw"]


def test_recency_weights_are_used(cfg):
    X, y = _xy(cfg)
    m = models.GBM("lightgbm", {**cfg["models"]["lightgbm"], "n_estimators": 20, "halflife_days": 30})
    m.fit(X[X["day"] >= "2021-01-15"], y[X["day"] >= "2021-01-15"].to_numpy())
    assert m.halflife == 30 and m.model is not None


def test_rolling_offset_removes_a_constant_gap():
    idx = pd.date_range("2024-01-01", periods=60 * 24, freq="1h", tz="UTC")
    day = idx.tz_localize(None).normalize()
    y = 50000 + 1000 * np.sin(np.arange(len(idx)))
    pred = pd.DataFrame({"day": day, "hour": idx.hour, "y": y, "f": y - 1200 - 100 * (idx.hour == 3)}, index=idx)
    adj = conformal.rolling_offset(pred, "f", window=14)
    late = pred["day"] >= pd.Timestamp("2024-01-20")
    np.testing.assert_allclose(adj[late], pred.loc[late, "y"], atol=1e-6)
    assert adj[pred["day"] < pd.Timestamp("2024-01-10")].isna().all()
