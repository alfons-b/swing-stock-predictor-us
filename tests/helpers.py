"""Fixture bersama: dataset contoh kecil + config SQLite sementara (cepat, deterministik, tanpa internet)."""
from __future__ import annotations

import copy
import os
import tempfile
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("YAHOO_ENABLED", "false")

from app.config import load_config  # noqa: E402
from app.data.sample import generate_sample_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START, END = "2017-01-02", "2020-12-31"


@lru_cache(maxsize=1)
def sample_dir() -> str:
    d = Path(tempfile.mkdtemp(prefix="ssp_sample_")) / "sample"
    generate_sample_dataset(d, n_tickers=14, start=START, end=END, seed=11)
    return str(d)


def make_cfg(db_url: str | None = None, as_of: str | None = None, **over) -> dict:
    cfg = copy.deepcopy(load_config(ROOT / "config"))
    tmp = Path(tempfile.mkdtemp(prefix="ssp_cfg_"))
    cfg["_root"] = str(tmp)
    cfg["database"]["url"] = db_url or f"sqlite:///{tmp / 'test.db'}"
    cfg["market_data"]["providers"] = [{"name": "csv", "type": "csv", "enabled": True, "priority": 1, "dir": sample_dir()}]
    cfg["market_data"]["retry"] = {"attempts": 2, "backoff_seconds": 0}
    cfg["universe"]["sources"] = ["csv_file"]
    cfg["calendar"]["holidays_file"] = str(ROOT / "config" / "holidays.yaml")
    cfg["data"]["start_date"] = START
    cfg["quality"]["min_history_days"] = 120
    cfg["liquidity"].update({"MIN_AVG_TRADING_VALUE": 1e6, "MIN_MEDIAN_TRADING_VALUE": 5e5, "MIN_AVG_VOLUME": 1e4})
    # tanpa internet & deterministik: sumber riset online (FINRA, SEC EDGAR) tidak dipakai di test
    cfg["foreign_flow"]["providers"] = [{"name": "csv_foreign_flow", "type": "csv", "enabled": True, "priority": 2,
                                         "dir": str(tmp / "foreign_flow_csv")}]
    cfg["fundamentals"]["providers"] = [p for p in cfg["fundamentals"]["providers"] if p.get("type") != "sec_edgar"]
    cfg["split"].update({"test_months": 4, "validation_folds": 1, "fold_months": 6, "warmup_trading_days": 220})
    cfg["model"].update({"candidates": ["logistic"], "ensemble_size": 1, "classifier": "auto"})
    cfg["backtest"]["weekly_validation_folds"] = 1
    cfg["storage"].update({"model_backend": "database", "report_backend": "database"})
    cfg["pipeline"]["scan_lookback_trading_days"] = 300
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    if as_of:
        cfg["_as_of"] = as_of
    return cfg


def make_ctx(cfg):
    from app.pipeline.context import AppContext
    return AppContext.create(cfg)


def db_urls() -> list[str]:
    """SQLite selalu; PostgreSQL bila TEST_POSTGRES_URL diset (GitHub Actions service container)."""
    urls = ["sqlite:///:memory:"]
    if os.environ.get("TEST_POSTGRES_URL"):
        urls.append(os.environ["TEST_POSTGRES_URL"])
    return urls
