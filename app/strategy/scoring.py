"""Multi-factor score (S) dan gabungan final (V).

Technical score = Σ bobot × komponen (0-100). Final score = gabungan Technical, ML,
Fundamental, Sentiment; komponen tanpa data dikeluarkan dan bobot dinormalisasi ulang
(tidak mengarang skor netral untuk data yang tidak ada).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.fundamental_sentiment import fundamental_score, sentiment_score

REGIME_SCORE = {"STRONG_BULL": 100, "BULL": 80, "NEUTRAL": 55, "BEAR": 30, "STRONG_BEAR": 10}


def _c(x):
    return np.clip(x, 0, 1)


def component_scores(df: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["score_trend"] = 100 * pd.concat([
        (c > df["ema20"]).astype(float), (df["ema20"] > df["ema50"]).astype(float),
        (df["ema50"] > df["sma200"]).astype(float), (df["f_ema20_slope5"] > 0).astype(float),
        pd.Series(_c(df["adx"] / 40), index=df.index) * (df["plus_di"] > df["minus_di"])], axis=1).mean(axis=1)
    rsi_bell = _c(1 - (df["rsi14"] - 62).abs() / 30)
    out["score_momentum"] = 100 * pd.concat([
        pd.Series(rsi_bell, index=df.index), (df["macd_hist"] > 0).astype(float),
        (df["f_macd_hist_slope"] > 0).astype(float), pd.Series(_c(df["f_roc10"] / 0.10 + 0.5), index=df.index)],
        axis=1).mean(axis=1)
    out["score_volume"] = 100 * pd.concat([
        pd.Series(_c((df["volume_ratio"] - 0.8) / 1.5), index=df.index), (df["f_obv_slope"] > 0).astype(float),
        pd.Series(_c((df["f_value_mom"] + 0.2) / 0.6), index=df.index)], axis=1).mean(axis=1)
    out["score_relative_strength"] = 100 * df[["f_rs20_rank", "f_rs60_rank"]].mean(axis=1)
    out["score_setup_quality"] = setups["setup_score"]
    out["score_market_regime"] = pd.to_numeric(df["market_regime"].astype(object).map(REGIME_SCORE), errors="coerce")
    out["score_sector_strength"] = df["sector_score"]
    out["score_liquidity"] = df["liquidity_score"]
    return out          # NaN = komponen tidak tersedia (mis. sektor tanpa cukup anggota) — BUKAN nol


def technical_score(comp: pd.DataFrame, weights: dict, min_coverage: float = 0.7) -> pd.Series:
    """Rata-rata berbobot komponen yang TERSEDIA (bobot dinormalisasi ulang).

    Sebelumnya komponen hilang diisi 0 sehingga emiten tanpa data sektor/regime dihukum (bias sistematis).
    Bila bobot komponen tersedia < min_coverage → NaN (tidak bisa dinilai)."""
    num = sum(weights[k] * comp[f"score_{k}"].fillna(0.0) for k in weights)
    den = sum(weights[k] * comp[f"score_{k}"].notna() for k in weights)
    total = sum(weights.values()) or 1.0
    return (num / den.replace(0, np.nan)).where(den / total >= min_coverage)


def ml_score(pred: pd.DataFrame, horizon: int) -> pd.Series:
    """Skor absolut (bukan ranking relatif) — hari yang buruk tetap menghasilkan skor rendah."""
    base = pred["base_rate_bullish"].fillna(1 / 3) if "base_rate_bullish" in pred else 1 / 3
    p = _c((pred["probability_bullish"] - base) / 0.15)  # lift +15pp di atas base rate = skor penuh
    er = pred.get(f"expected_return_{horizon}d", pd.Series(0.0, index=pred.index))
    r = _c((er + 0.02) / 0.07)
    tp = _c(pred.get("prob_hit_tp", pd.Series(0.25, index=pred.index)) / 0.5)
    return 100 * (0.5 * p + 0.3 * r + 0.2 * tp)


def final_score(df: pd.DataFrame, tech: pd.Series, ml: pd.Series, cfg: dict) -> pd.DataFrame:
    w = cfg["scoring"]["final_weights"]
    parts = {"technical": tech, "ml": ml, "fundamental": fundamental_score(df), "sentiment": sentiment_score(df)}
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for k, s in parts.items():
        avail = s.notna()
        num += np.where(avail, w.get(k, 0) * s.fillna(0), 0)
        den += np.where(avail, w.get(k, 0), 0)
    out = pd.DataFrame({f"{k}_score": v for k, v in parts.items()}, index=df.index)
    out["final_score"] = (num / den.replace(0, np.nan)).round(1)
    return out
