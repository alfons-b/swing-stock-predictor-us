"""Orkestrasi feature engineering → satu DataFrame panel (ticker × date)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import get
from app.features.fundamental_sentiment import merge_fundamentals, merge_news
from app.market.market_regime import compute_breadth, compute_market_regime
from app.market.sector_strength import add_relative_strength, compute_sector_strength
from app.features.technical import compute_stock_features
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

FEATURE_ABS_MAX = 1e6  # rasio keuangan > 1 juta tidak bermakna; juga menjaga nilai tetap aman untuk float32


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Jaring pengaman terakhir: ±inf → NaN dan nilai ekstrem dipotong ke ±FEATURE_ABS_MAX untuk semua kolom f_*."""
    cols = [c for c in df.columns if c.startswith("f_")]
    if not cols:
        return df
    # kolom per kolom → puncak memori ±1 kolom (dataset production ±1,7 jt baris × 85 fitur)
    n_extreme, fixed = 0, {}
    for c in cols:
        arr = df[c].to_numpy(dtype=float, copy=True)
        with np.errstate(invalid="ignore"):
            arr[~np.isfinite(arr)] = np.nan
            big = np.abs(arr) > FEATURE_ABS_MAX
        if big.any():
            n_extreme += int(big.sum())
            np.clip(arr, -FEATURE_ABS_MAX, FEATURE_ABS_MAX, out=arr)
            fixed[c] = arr
    if n_extreme:
        log.warning("%d nilai fitur ekstrem (>|%g|) dipotong di %d kolom", n_extreme, FEATURE_ABS_MAX, len(fixed))
        for c, arr in fixed.items():  # hanya kolom yang benar-benar berubah yang ditulis ulang
            df[c] = arr
    return df


REGIME_FEATURES = ["idx_ret5", "idx_ret20", "idx_ret60", "idx_dist_sma50", "idx_dist_sma200", "idx_vol20",
                   "idx_vol_pct", "breadth_above_sma50", "breadth_adv_ratio", "regime_score", "regime_code"]
SECTOR_FEATURES = ["sector_ret5", "sector_ret20", "sector_ret60", "sector_breadth", "sector_rs20", "sector_score"]


def build_features(market: dict, cfg: dict) -> pd.DataFrame:
    prices, universe, index = market["prices"], market["universe"], market["index"]
    df = prices.merge(universe[["ticker", "name", "sector", "subsector"]], on="ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown")

    parts = [compute_stock_features(g, cfg) for _, g in df.groupby("ticker", sort=False)]
    df = pd.concat(parts, ignore_index=True)
    log.info("Fitur per-saham: %d baris, %d ticker", len(df), df["ticker"].nunique())

    breadth = compute_breadth(df)
    regime = compute_market_regime(index, breadth, cfg)
    df = df.merge(regime.drop(columns=["breadth_above_sma200", "breadth_new_high20"], errors="ignore"),
                  on="date", how="left")
    df["market_regime"] = df["market_regime"].fillna("NEUTRAL")

    sector = compute_sector_strength(df)
    df = df.merge(sector, on=["date", "sector"], how="left")
    df = add_relative_strength(df)

    df = merge_fundamentals(df, market.get("fundamentals"))
    df = merge_news(df, market.get("news"))

    for c in REGIME_FEATURES + SECTOR_FEATURES:
        if c in df:
            df[f"f_{c}"] = df[c]
    if "news_sent5" in df:
        df["f_news_sent5"] = df["news_sent5"]
        df["f_news_count5"] = df["news_count5"]
    df = add_liquidity(df, cfg)
    df = sanitize_features(df.replace([np.inf, -np.inf], np.nan))
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def add_liquidity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lq = cfg.get("liquidity", {})
    ok = (df["avg_value20"] >= lq.get("MIN_AVG_TRADING_VALUE", 0)) & \
         (df["median_value20"] >= lq.get("MIN_MEDIAN_TRADING_VALUE", 0)) & \
         (df["avg_volume20"] >= lq.get("MIN_AVG_VOLUME", 0)) & \
         (df["active_days20"] >= lq.get("MIN_ACTIVE_TRADING_DAYS", 0))
    df["is_liquid"] = ok
    df["liquidity_score"] = 100 * df["f_log_value20"].groupby(df["date"]).rank(pct=True)
    df["is_tradeable"] = df["is_liquid"] & ~df["is_suspended"] & \
        (df["days_listed"] >= get(cfg, "quality.min_history_days", 250)) & \
        (df["close"] >= get(cfg, "quality.min_price", 50))
    return df


def feature_columns(df: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    cols = sorted(c for c in df.columns if c.startswith("f_"))
    excl = set(get(cfg or {}, "features.exclude", []) or [])
    return [c for c in cols if c not in excl]
