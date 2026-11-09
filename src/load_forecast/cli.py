"""`flf` command line."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from load_forecast.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flf", description="Day-ahead forecasting of French electricity load.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=None, help="YAML config (optional).")

    p_fetch = sub.add_parser("fetch", parents=[common], help="Download / update RTE and Open-Meteo data.")
    p_fetch.add_argument("--sources", nargs="+", choices=["rte", "era5", "day1", "day2"])

    p_bt = sub.add_parser("backtest", parents=[common], help="Monthly-refit backtest (runs/backtest_<tag>.parquet)")
    p_bt.add_argument("--models", nargs="+", help="naive_week, lightgbm, catboost, lightgbm_q05, ...")
    p_bt.add_argument("--weather", choices=["day1", "day2", "obs"], help="Weather used for the target day.")
    p_bt.add_argument("--tag", help="Output name (default: the weather source).")
    p_bt.add_argument("--no-debias", action="store_true", help="Feed the raw forecasts (no forecast - ERA5 offset).")
    p_bt.add_argument("--append", action="store_true", help="Add the models to an existing backtest file.")
    p_bt.add_argument("--start", help="First day of the test period (default: config).")
    p_bt.add_argument("--end", help="Last day of the test period (default: config).")

    p_rep = sub.add_parser("report", parents=[common], help="Tables and figures from the saved backtests.")
    p_rep.add_argument("--out", type=Path, default=Path("docs/assets"))

    p_live = sub.add_parser("live", parents=[common], help="Forecast tomorrow, update the archive and the dashboard.")
    p_live.add_argument("--archive", type=Path, default=Path("archive"), help="Folder of the forecast archive.")
    p_live.add_argument("--site", type=Path, default=Path("site"), help="Where the dashboard is written.")
    p_live.add_argument("--no-fetch", action="store_true", help="Use the data already on disk.")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    overrides = {}
    if getattr(args, "no_debias", False):
        overrides["weather"] = {"debias": False}
    period = {k: getattr(args, k, None) for k in ("start", "end")}
    if any(period.values()):
        overrides["backtest"] = {k: v for k, v in period.items() if v}
    cfg = load_config(args.config, overrides)

    if args.command == "fetch":
        from load_forecast import data

        data.fetch(cfg, args.sources)
    elif args.command == "backtest":
        from load_forecast import backtest, metrics

        weather = args.weather or cfg["backtest"]["weather"]
        tag = args.tag or weather
        pred = backtest.run(cfg, args.models, weather)
        if args.append:
            old = backtest.read(cfg, tag)
            for col in pred.columns.difference(old.columns):
                old[col] = pred[col]
            pred = old
        path = backtest.save(pred, cfg, tag)
        cols = [c for c in pred.columns if c not in ("day", "hour", "y")]
        print(metrics.table(pred, cols).round(3).to_string())
        print(f"saved {path}")
    elif args.command == "report":
        from load_forecast import report

        print(report.write_all(cfg, args.out))
    elif args.command == "live":
        import pandas as pd

        from load_forecast import live, site

        now = pd.Timestamp.now(tz=cfg["timezone"])
        _, replay = live.forecast(cfg, args.archive, fetch=not args.no_fetch, now=now)
        path = site.build(live.scored_archive(cfg, args.archive), args.site, cfg, now, replay)
        print(f"dashboard: {path}")


if __name__ == "__main__":
    main()
