"""Setup detector (I). Rule-based, transparan, semua threshold dari config.

Setiap setup menghasilkan: valid (bool), score (0-100), invalidation level.
Baris dapat memenuhi beberapa setup; dipilih yang skornya tertinggi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SETUP_NAMES = ["BREAKOUT", "BREAKOUT_VOLUME", "PULLBACK_UPTREND", "EMA20_PULLBACK", "SUPPORT_BOUNCE",
               "TREND_CONTINUATION", "MOMENTUM_CONTINUATION", "VCP_BREAKOUT", "HH_HL", "BASE_BREAKOUT"]
BREAKOUT_SETUPS = {"BREAKOUT", "BREAKOUT_VOLUME", "VCP_BREAKOUT", "BASE_BREAKOUT"}
PULLBACK_SETUPS = {"PULLBACK_UPTREND", "EMA20_PULLBACK", "SUPPORT_BOUNCE"}


def _c(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def detect_setups(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg.get("setups", {})
    c, o, low = df["close"], df["open"], df["low"]
    vr = df["volume_ratio"].fillna(0)
    uptrend = (c > df["ema50"]) & (df["ema20"] > df["ema50"])
    long_up = uptrend & (df["ema50"] > df["sma200"])
    adx_ok = (df["adx"] >= s.get("adx_trend_min", 20)) & (df["plus_di"] > df["minus_di"])
    trend_q = (_c(df["adx"] / 40) * 0.5 + uptrend.astype(float) * 0.5)
    regime_ok = df["market_regime"] != "STRONG_BEAR"
    atr = df["atr14"]

    valid, score, inval = {}, {}, {}
    bo = (c > df["res20"]) & (c > df["ema50"])
    valid["BREAKOUT"] = bo
    score["BREAKOUT"] = 45 + 25 * trend_q + 15 * _c((vr - 1) / 1.5) + 15 * df["f_close_pos"].fillna(0)
    inval["BREAKOUT"] = df["res20"] - 0.5 * atr

    bv = bo & (vr > s.get("volume_ratio_threshold", 1.5)) & regime_ok
    valid["BREAKOUT_VOLUME"] = bv
    score["BREAKOUT_VOLUME"] = 55 + 20 * trend_q + 15 * _c((vr - 1.5) / 2) + 10 * df["f_close_pos"].fillna(0)
    inval["BREAKOUT_VOLUME"] = df["res20"] - 0.5 * atr

    pb = long_up & (df["ret5"] < 0) & df["rsi14"].between(38, 55) & (c > df["ema50"])
    valid["PULLBACK_UPTREND"] = pb
    score["PULLBACK_UPTREND"] = 50 + 25 * trend_q + 25 * _c(1 - (df["rsi14"] - 38) / 17)
    inval["PULLBACK_UPTREND"] = np.minimum(df["swing_low10"], df["ema50"]) - 0.2 * atr

    tol = s.get("pullback_tolerance", 0.015)
    ep = (low <= df["ema20"] * (1 + tol)) & (c > df["ema20"]) & (df["ema20"] > df["ema50"]) & (df["f_ema20_slope5"] > 0)
    valid["EMA20_PULLBACK"] = ep
    score["EMA20_PULLBACK"] = 50 + 25 * trend_q + 15 * df["f_close_pos"].fillna(0) + 10 * (c > o)
    inval["EMA20_PULLBACK"] = np.minimum(df["swing_low10"], df["ema20"] - atr)

    stol = s.get("support_tolerance", 0.02)
    sb = (low <= df["sup20"] * (1 + stol)) & (c > df["sup20"]) & (c > o) & (df["f_close_pos"] > 0.6) & (c > df["sma200"])
    valid["SUPPORT_BOUNCE"] = sb
    score["SUPPORT_BOUNCE"] = 45 + 20 * df["f_close_pos"].fillna(0) + 20 * _c(vr / 2) + 15 * (df["ema50"] > df["sma200"])
    inval["SUPPORT_BOUNCE"] = df["sup20"] - 0.3 * atr

    tc = (c > df["ema20"]) & long_up & adx_ok
    valid["TREND_CONTINUATION"] = tc
    score["TREND_CONTINUATION"] = 45 + 30 * _c(df["adx"] / 45) + 25 * _c(df["f_rs20_rank"].fillna(0.5))
    inval["TREND_CONTINUATION"] = df["ema50"] - 0.2 * atr

    mc = (df["f_roc10"] > s.get("momentum_roc_min", 0.05)) & df["rsi14"].between(55, 75) & (df["macd_hist"] > 0) & \
         (df["f_macd_hist_slope"] > 0) & (df["f_rs20_rank"] > 0.7)
    valid["MOMENTUM_CONTINUATION"] = mc
    score["MOMENTUM_CONTINUATION"] = 50 + 25 * _c(df["f_rs20_rank"].fillna(0)) + 25 * _c(df["f_roc10"] / 0.15)
    inval["MOMENTUM_CONTINUATION"] = df["swing_low10"] - 0.2 * atr

    prev_bbw = df.groupby("ticker", sort=False)["bb_width_pct"].shift(1)
    vcp = bo & (prev_bbw < s.get("vcp_bbw_pct_max", 0.25)) & (vr > 1.2)
    valid["VCP_BREAKOUT"] = vcp
    score["VCP_BREAKOUT"] = 60 + 20 * _c(1 - prev_bbw.fillna(1) / 0.25) + 20 * _c((vr - 1.2) / 1.5)
    inval["VCP_BREAKOUT"] = df["res20"] - 0.5 * atr

    hh = (df["higher_high"] > 0) & (df["higher_low"] > 0) & (c > df["ema20"])
    valid["HH_HL"] = hh
    score["HH_HL"] = 45 + 30 * trend_q + 25 * _c(df["f_rs20_rank"].fillna(0.5))
    inval["HH_HL"] = df["swing_low10"] - 0.2 * atr

    base = (df["base_range"] < s.get("base_max_range", 0.18)) & (c > df["base_high"]) & (vr > 1.2)
    valid["BASE_BREAKOUT"] = base
    score["BASE_BREAKOUT"] = 55 + 25 * _c(1 - df["base_range"].fillna(1) / 0.18) + 20 * _c((vr - 1.2) / 1.5)
    inval["BASE_BREAKOUT"] = df["base_low"].where(df["base_low"] > c * 0.85, df["base_high"] - atr)

    V = pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in valid.items()})
    S = pd.DataFrame({k: pd.Series(score[k], index=df.index).clip(0, 100) for k in SETUP_NAMES}).where(V)
    I = pd.DataFrame({k: pd.Series(inval[k], index=df.index) for k in SETUP_NAMES})
    out = pd.DataFrame(index=df.index)
    out["setup_valid"] = V.any(axis=1)
    best = S.fillna(-1).idxmax(axis=1)
    out["setup_type"] = best.where(out["setup_valid"], "NONE")
    out["setup_score"] = S.max(axis=1).fillna(0.0)
    out["invalidation_level"] = [I.at[i, b] if b != "NONE" else np.nan for i, b in zip(df.index, out["setup_type"])]
    out["setups_all"] = V.apply(lambda r: ",".join(r.index[r.to_numpy()]), axis=1) if len(V) < 5000 else ""
    # "forming": calon watchlist — mendekati resistance dengan volatilitas menyempit, atau uptrend tanpa trigger
    near_res = (df["res20"] / c - 1).between(0, 0.03) & uptrend
    out["setup_forming"] = ~out["setup_valid"] & (near_res | (long_up & (df["bb_width_pct"] < 0.3)))
    out["is_low_quality_breakout"] = out["setup_type"].isin(BREAKOUT_SETUPS) & \
        (out["setup_score"] < s.get("low_quality_breakout_score", 60))
    return out
