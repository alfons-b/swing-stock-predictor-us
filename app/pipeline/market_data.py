"""Memuat data pasar dari DATABASE (bukan file) → cleaner → feature engine. Dipakai daily, train, backtest."""
from __future__ import annotations

import time

import pandas as pd

from app.config import get
from app.data.cleaner import clean_prices
from app.database.repository import Repository
from app.features.builder import build_features
from app.models.labels import add_labels
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def load_market(cfg: dict, repo: Repository, start=None, end=None) -> dict:
    prices = repo.load_prices(start=start, end=end)
    stocks = repo.stocks()
    uni = stocks.rename(columns={})[["ticker", "name", "sector", "subsector", "listing_date", "delisting_date", "board"]].copy()
    uni["sector"] = uni["sector"].fillna("Unknown").replace("", "Unknown").astype(object)
    uni["name"] = uni["name"].fillna(uni["ticker"]).astype(object)
    uni["subsector"] = uni["subsector"].fillna("Unknown").astype(object)
    uni["ticker"] = uni["ticker"].astype(object)
    ca = repo.load_actions()
    index = repo.load_index(get(cfg, "data.index_id", "COMPOSITE"), start=start)
    if end is not None:
        index = index[index["date"] <= pd.Timestamp(end)]
    if prices.empty or index.empty:
        raise RuntimeError("Database belum berisi data harga/S&P 500 — jalankan `python main.py setup` atau `update-data`")
    clean, notes = clean_prices(prices, uni, ca, cfg, provider_adjusted=False)
    return {"prices": clean, "index": index, "universe": uni, "corporate_actions": ca,
            "fundamentals": pd.DataFrame(), "news": pd.DataFrame(), "cleaner_notes": notes}


def build_dataset(cfg: dict, repo: Repository, lookback_trading_days: int | None = None, labels: bool = True,
                  end=None) -> pd.DataFrame:
    t = time.time()
    start = None
    if lookback_trading_days:
        dates = repo.index_dates(get(cfg, "data.index_id", "COMPOSITE"))
        if end is not None:
            dates = [d for d in dates if d <= pd.Timestamp(end)]
        if len(dates) > lookback_trading_days:
            start = dates[-lookback_trading_days]
    market = load_market(cfg, repo, start=start, end=end)
    df = build_features(market, cfg)
    if labels:
        df = add_labels(df, cfg)
    log.info("Dataset: %d baris, %d emiten, %s..%s (%.1fs)", len(df), df["ticker"].nunique(),
             df["date"].min().date(), df["date"].max().date(), time.time() - t)
    return df
