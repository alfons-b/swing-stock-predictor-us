"""Pipeline sinyal (T) — dipakai IDENTIK oleh `scan` (live) dan backtest (historis).

generate_signals: fitur + prediksi model → setup, entry, SL, TP, RR, sizing, skor, filter, keputusan.
rank_signals    : per tanggal, ambil BUY teratas dengan batas jumlah per regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.entry import compute_entry
from app.strategy.filters import apply_filters
from app.risk.position_sizing import compute_position_size
from app.strategy.scoring import component_scores, final_score, ml_score, technical_score
from app.strategy.setups import detect_setups
from app.strategy.stop_loss import compute_stops
from app.strategy.take_profit import compute_targets

PRED_COLS = ["probability_bearish", "probability_neutral", "probability_bullish", "prob_dispersion"]


def generate_signals(df: pd.DataFrame, pred: pd.DataFrame, cfg: dict, portfolio_value: float | None = None) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    h = cfg["labels"]["SWING_HORIZON"]
    setups = detect_setups(df, cfg)
    entry = compute_entry(df, setups, cfg)
    stops = compute_stops(df, setups, entry, cfg)
    targets = compute_targets(df, entry, stops, cfg)
    sizing = compute_position_size(df, entry, stops, cfg, portfolio_value)
    comp = component_scores(df, setups)
    tech = technical_score(comp, cfg["scoring"]["technical_weights"])
    ml = ml_score(pred, h)
    scores = final_score(df, tech, ml, cfg)
    filt = apply_filters(df, setups, pred, stops, targets, sizing, scores, cfg)
    out = pd.concat([df, pred[[c for c in pred.columns if c not in df.columns]], setups, entry, stops, targets,
                     sizing, comp, scores, filt], axis=1)
    rr_c = np.clip((out["risk_reward"] - 1) / 2, 0, 1)
    out["rank_score"] = (0.85 * out["final_score"] + 15 * rr_c) * (1 - out["prob_dispersion"].fillna(0))
    return out


def rank_signals(sig: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    top_n = cfg["scoring"]["TOP_N_STOCKS"]
    rules = cfg.get("regime", {}).get("rules", {})
    buys = sig[sig["decision"] == "BUY"].sort_values(["date", "rank_score"], ascending=[True, False])
    if buys.empty:  # NO TRADE. (Juga mencegah bug dtype: map() pada Series kosong ber-dtype str/pyarrow.)
        return buys.assign(rank=pd.Series(dtype="int64")).reset_index(drop=True)
    buys = buys.assign(rank=buys.groupby("date").cumcount() + 1)
    cap = pd.to_numeric(buys["market_regime"].astype(object).map(
        lambda r: rules.get(r, {}).get("max_recommendations", top_n)), errors="coerce").fillna(top_n).clip(upper=top_n)
    return buys[buys["rank"].to_numpy() <= cap.to_numpy()].reset_index(drop=True)
