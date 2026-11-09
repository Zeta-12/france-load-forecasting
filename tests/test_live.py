import pandas as pd

from load_forecast import live, rte, site, weather
from tests.conftest import synthetic


def test_live_forecast_and_dashboard(cfg, tmp_path, monkeypatch):
    load_df, wx = synthetic("2022-12-01", "2023-04-12")
    now = pd.Timestamp("2023-04-10 10:00", tz="Europe/Paris")
    # real-time data goes a bit beyond the cut-off, RTE's J-1 covers tomorrow
    load_df.loc[load_df.index > now.tz_convert("UTC"), "load"] = float("nan")
    monkeypatch.setattr(rte, "load_hourly", lambda data_dir: load_df)
    monkeypatch.setattr(rte, "load_realtime", lambda data_dir: load_df)
    monkeypatch.setattr(weather, "load_national", lambda data_dir, source: wx)
    monkeypatch.setattr(weather, "fetch_forecast", lambda *a, **k: None)
    monkeypatch.setattr(weather, "national", lambda long: wx)
    cfg = {**cfg, "history_start": "2022-12-15", "train": {"exclude": []}}
    cfg["models"] = {
        **cfg["models"],
        "lightgbm": {**cfg["models"]["lightgbm"], "n_estimators": 100},
        "catboost": {**cfg["models"]["catboost"], "iterations": 100},
    }
    cfg["conformal"] = {**cfg["conformal"], "window_days": 21, "replay_days": 40}

    archive = tmp_path / "archive"
    fc, replay = live.forecast(cfg, archive, fetch=False, now=now)
    assert len(fc) == 24
    assert (archive / "forecasts" / "2023-04-11.csv").exists()
    assert (fc["lo"] < fc["forecast"]).all() and (fc["forecast"] < fc["hi"]).all()
    assert fc["rte_j1"].notna().all()

    # a past forecast whose day is now known
    old = fc.copy()
    old["time"] = (pd.to_datetime(old["time"]) - pd.Timedelta(days=3)).dt.strftime("%Y-%m-%dT%H:%MZ")
    old.to_csv(archive / "forecasts" / "2023-04-08.csv", index=False)
    scored = live.scored_archive(cfg, archive)
    assert scored.loc[scored["day"] == "2023-04-08", "actual"].notna().all()
    assert scored.loc[scored["day"] == "2023-04-11", "actual"].isna().all()
    # actuals are frozen once archived, later revisions of the source don't change them
    frozen = (archive / "actuals.csv").read_text()
    load_df["load"] += 1000
    live.scored_archive(cfg, archive)
    assert (archive / "actuals.csv").read_text() == frozen

    assert replay["actual"].notna().all() and replay["day"].max() == "2023-04-09"
    page = site.build(scored, tmp_path / "site", cfg, now, replay)
    html = page.read_text(encoding="utf-8")
    assert "Forecast for" in html and "2023-04-11" in html
    assert site.replay_scores(replay)["days"] >= 28
    assert site.scores(scored, 0.1)[0]["days"] == 1
