import numpy as np
import pandas as pd

from load_forecast import conformal
from load_forecast.metrics import interval_metrics


def _pred(n_days=200, shift_at=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    day = idx.tz_localize(None).normalize()
    noise = rng.normal(0, 1000, len(idx))
    if shift_at is not None:
        noise[day >= pd.Timestamp(shift_at)] *= 3
    y = 50000 + noise
    return pd.DataFrame({"day": day, "hour": idx.hour, "y": y, "f": 50000.0}, index=idx)


def test_rolling_coverage_iid():
    pred = _pred()
    iv, _ = conformal.rolling(pred, "f", alpha=0.1, window=28)
    ok = iv["lo"].notna()
    m = interval_metrics(pred["y"][ok], iv["lo"][ok], iv["hi"][ok], 0.1)
    assert abs(m["coverage"] - 0.9) < 0.02


def test_uses_only_days_up_to_d_minus_2():
    pred = _pred()
    iv, _ = conformal.rolling(pred, "f", alpha=0.1, window=28, by_hour=True)
    d = pd.Timestamp("2024-03-10")
    p2 = pred.copy()
    p2.loc[p2["day"] >= d - pd.Timedelta(days=1), "y"] += 1e6
    iv2, _ = conformal.rolling(p2, "f", alpha=0.1, window=28, by_hour=True)
    rows = pred["day"] == d
    pd.testing.assert_frame_equal(iv[rows], iv2[rows])


def test_aci_recovers_coverage_after_a_shift():
    pred = _pred(n_days=300, shift_at="2024-05-01")
    after = pred["day"] >= pd.Timestamp("2024-07-01")
    static, _ = conformal.rolling(pred, "f", alpha=0.1, window=28)
    adaptive, levels = conformal.rolling(pred, "f", alpha=0.1, window=28, gamma=0.05)
    cov = interval_metrics(pred["y"][after], adaptive["lo"][after], adaptive["hi"][after], 0.1)["coverage"]
    assert abs(cov - 0.9) < 0.03
    # right after the shift ACI lowers its level to widen the intervals
    assert levels["2024-05-05":"2024-05-20"].min() < 0.1
    assert static["lo"].notna().sum() == adaptive["lo"].notna().sum()


def test_cqr_widens_a_too_narrow_band():
    pred = _pred()
    pred["q_lo"] = 50000 - 500.0
    pred["q_hi"] = 50000 + 500.0
    iv = conformal.cqr(pred, "q_lo", "q_hi", alpha=0.1, window=28)
    ok = iv["lo"].notna()
    raw = interval_metrics(pred["y"][ok], pred["q_lo"][ok], pred["q_hi"][ok], 0.1)["coverage"]
    cal = interval_metrics(pred["y"][ok], iv["lo"][ok], iv["hi"][ok], 0.1)["coverage"]
    assert raw < 0.5
    assert abs(cal - 0.9) < 0.02
