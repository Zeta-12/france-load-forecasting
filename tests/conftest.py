import numpy as np
import pandas as pd
import pytest

from load_forecast.config import load_config


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"data_dir": str(tmp_path / "data"), "output_dir": str(tmp_path / "runs")})


def synthetic(start="2023-01-01", end="2023-04-30", seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly load driven by temperature, with daily and weekly patterns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1h", tz="UTC", name="time")
    t = np.arange(len(idx))
    temp = 8 + 6 * np.sin(2 * np.pi * t / (24 * 365)) + 4 * np.sin(2 * np.pi * (t - 15) / 24) + rng.normal(0, 1, len(t))
    daily = 5000 * np.sin(2 * np.pi * (t - 8) / 24)
    weekend = np.where(idx.day_of_week >= 5, -6000, 0)
    load = 55000 + daily + weekend - 1500 * (temp - 10) + rng.normal(0, 500, len(t))
    load_df = pd.DataFrame({"load": load, "rte_j1": load + rng.normal(0, 800, len(t)), "rte_j": load}, index=idx)
    wx = pd.DataFrame(
        {"temp": temp, "cloud": rng.uniform(0, 100, len(t)), "solar": 0.0, "wind": rng.uniform(0, 30, len(t))},
        index=idx,
    )
    return load_df, wx


@pytest.fixture
def toy():
    return synthetic()
