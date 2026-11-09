from pathlib import Path

import pandas as pd
import pytest
import requests

from load_forecast import data, weather


def fake_get(url, params):
    times = pd.date_range(params["start_date"], f"{params['end_date']} 23:00", freq="1h")
    hourly = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
    for name in params["hourly"].split(","):
        hourly[name] = [1.0] * len(times)
    return {"hourly": hourly}


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)


def test_chunks_cover_the_range_once():
    start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2017-03-10")
    parts = data.chunks(start, end, days=365)
    assert parts[0][0] == start and parts[-1][1] == end
    for (_, b), (c, _) in zip(parts, parts[1:], strict=False):
        assert c == b + pd.Timedelta(days=1)


def test_interrupted_download_resumes(cfg, monkeypatch, quiet):
    cfg = {**cfg, "history_start": "2023-01-01"}
    calls = {"n": 0}

    def flaky(url, params):
        calls["n"] += 1
        if calls["n"] == 2 * len(weather.REGIONS) + 3:
            raise RuntimeError("open-meteo down")
        return fake_get(url, params)

    monkeypatch.setattr(weather, "_get", flaky)
    today = pd.Timestamp("2026-01-15")
    with pytest.raises(RuntimeError):
        data.fetch(cfg, ["era5"], today=today)
    store = Path(cfg["data_dir"]) / "raw" / "weather" / "era5.parquet"
    # the first two years were saved before the failure
    saved = pd.read_parquet(store)["time"].max()
    assert pd.Timestamp("2024-11-01", tz="UTC") < saved < pd.Timestamp("2025-01-15", tz="UTC")

    monkeypatch.setattr(weather, "_get", fake_get)
    data.fetch(cfg, ["era5"], today=today)
    df = pd.read_parquet(store)
    assert df["time"].max() == pd.Timestamp("2026-01-15 23:00", tz="UTC")
    assert not df.duplicated(["time", "region"]).any()
    assert set(df["region"]) == {r[0] for r in weather.REGIONS}


def test_timeouts_are_retried(monkeypatch, quiet):
    class Ok:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    attempts = []

    def get(url, params, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ReadTimeout("read timed out")
        return Ok()

    monkeypatch.setattr(weather._session, "get", get)
    assert weather._get("https://example.org", {}) == {"ok": True}
    assert len(attempts) == 3


def test_pause_keeps_under_the_quota():
    # one region-year weighs ~26 calls, 12 regions in a row must stay under 600 calls per minute
    per_minute = 26 * 60 / (weather.pause_for(365) + 0.5)
    assert per_minute < 600
    assert weather.pause_for(3) < 0.2
