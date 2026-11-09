"""Point forecasters. They all take the feature table of features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_forecast.features import FEATURES, LOAD_FEATURES


class NaiveWeek:
    """Same hour one week before."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_d7"].to_numpy()


class GBM:
    """LightGBM / CatBoost regressor.

    anchor: fit y - X[anchor] instead of y
    relative: with an anchor, the other load features are given as differences to it too
    halflife_days: sample weights halving every `halflife_days` before the last training day
    """

    def __init__(self, name: str, params: dict, n_jobs: int = 4, seed: int = 0):
        self.name = name
        self.params = dict(params)
        self.anchor = self.params.pop("anchor", None)
        self.relative = self.params.pop("relative", False)
        self.halflife = self.params.pop("halflife_days", None)
        self.n_jobs = n_jobs
        self.seed = seed
        self.model = None

    def _make(self):
        if self.name.startswith("lightgbm"):
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                **self.params, n_jobs=self.n_jobs, force_col_wise=True, random_state=self.seed, verbose=-1
            )
        if self.name.startswith("catboost"):
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                **self.params, thread_count=self.n_jobs, random_seed=self.seed, verbose=0, allow_writing_files=False
            )
        raise ValueError(self.name)

    def _inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X[FEATURES]
        if self.anchor and self.relative:
            Z = Z.copy()
            ref = X[self.anchor].to_numpy()
            for col in LOAD_FEATURES:
                if col != self.anchor:
                    Z[col] = Z[col].to_numpy() - ref
        return Z

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model = self._make()
        if self.anchor:
            ref = X[self.anchor].to_numpy()
            ok = ~np.isnan(ref)
            X, y = X[ok], y[ok] - ref[ok]
        weight = None
        if self.halflife:
            age = (X["day"].max() - X["day"]).dt.days.to_numpy()
            weight = 0.5 ** (age / self.halflife)
        self.model.fit(self._inputs(X), y, sample_weight=weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict(self._inputs(X))
        return p + X[self.anchor].to_numpy() if self.anchor else p


def make(name: str, cfg: dict):
    if name == "naive_week":
        return NaiveWeek()
    if name.startswith("lightgbm_q"):
        # quantile regression, e.g. lightgbm_q05 -> 5 % quantile
        q = int(name.removeprefix("lightgbm_q")) / 100
        params = {**cfg["models"]["lightgbm"], **cfg["models"]["lightgbm_quantile"]}
        params.update(objective="quantile", alpha=q)
        return GBM(name, params, cfg["n_jobs"])
    if name == "lightgbm_abs":
        # ablation: same model on the absolute load
        params = {**cfg["models"]["lightgbm"], "anchor": None, "relative": False}
        return GBM(name, params, cfg["n_jobs"])
    return GBM(name, cfg["models"][name], cfg["n_jobs"])
