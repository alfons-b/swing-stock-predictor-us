"""Explainable output (X). Setiap alasan berasal dari nilai data aktual pada baris tersebut.

Kontribusi model memakai sensitivity lokal model-agnostic: fitur diganti median training
satu per satu, lalu dihitung perubahan probability_bullish. Bila `shap` terpasang dan
anggota ensemble berbasis tree, SHAP bisa dipakai sebagai alternatif.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REJECT_TEXT = {
    "illiquid": "Likuiditas di bawah batas minimum", "suspended": "Saham sedang suspensi / tidak ada transaksi",
    "short_history": "Riwayat perdagangan terlalu pendek (IPO baru)", "price_too_low": "Harga di bawah batas minimum",
    "corporate_action_risk": "Ada corporate action / lonjakan harga tak wajar dalam 10 hari terakhir",
    "fundamental_risk": "Risiko fundamental ekstrem (DER tinggi / ROE sangat negatif)",
    "extreme_volatility": "Volatilitas (ATR%) terlalu ekstrem", "setup_not_confirmed": "Belum ada setup yang terkonfirmasi",
    "low_quality_breakout_in_weak_regime": "Breakout berkualitas rendah saat regime pasar lemah",
    "stop_too_wide": "Jarak stop loss terlalu lebar", "risk_reward_below_min": "Risk/reward di bawah minimum",
    "probability_below_min": "Probabilitas bullish di bawah ambang (sudah disesuaikan regime)",
    "bearish_outweighs_bullish": "Probabilitas bearish >= bullish", "expected_return_not_positive": "Expected return tidak positif",
    "model_uncertainty_high": "Anggota ensemble tidak sepakat (uncertainty tinggi)",
    "position_too_small": "Ukuran posisi < 1 saham setelah batas risiko/likuiditas", "score_below_buy_min": "Skor akhir di bawah ambang BUY",
}

FEATURE_TEXT = {
    "f_rs20": "relative strength 20H vs S&P 500", "f_rs60": "relative strength 60H vs S&P 500", "f_ret20": "return 20H",
    "f_ret60": "return 60H", "f_dist_ema20": "jarak ke EMA20", "f_volume_ratio": "rasio volume", "f_rsi14": "RSI14",
    "f_macd_hist": "histogram MACD", "f_adx": "ADX", "f_atr_pct": "ATR%", "f_regime_score": "skor regime pasar",
    "f_sector_score": "skor sektor", "f_roc10": "ROC10", "f_dist_sma200": "jarak ke SMA200", "f_ema20_slope5": "slope EMA20",
}


def feature_label(f: str) -> str:
    if f in FEATURE_TEXT:
        return FEATURE_TEXT[f]
    name = f[2:] if f.startswith("f_") else f
    for k, v in (("dist_", "jarak ke "), ("ret", "return "), ("sma", "SMA"), ("ema", "EMA"), ("rsi", "RSI"),
                 ("idx_", "S&P 500 "), ("sector_", "sektor "), ("_", " ")):
        name = name.replace(k, v)
    return name.strip()


def _pct(x):
    return f"{x * 100:+.1f}%"


def build_reasoning(row: pd.Series) -> tuple[list[str], list[str]]:
    pros, cons = [], []
    c = row["close"]
    if row["ema20"] > row["ema50"] > row["sma200"] and c > row["ema20"]:
        pros.append("Struktur trend bullish: close > EMA20 > EMA50 > SMA200")
    elif c < row["sma200"]:
        cons.append(f"Close di bawah SMA200 ({_pct(c / row['sma200'] - 1)})")
    if c > row["ema20"]:
        pros.append(f"Close di atas EMA20 ({_pct(c / row['ema20'] - 1)})")
    if row.get("breakout20", 0) > 0:
        pros.append(f"Breakout resistance 20H di ${row['res20']:,.2f}")
    vr = row["volume_ratio"]
    if vr >= 1.5:
        pros.append(f"Volume {vr:.1f}x rata-rata 20H")
    elif vr < 0.8:
        cons.append(f"Volume lemah ({vr:.1f}x rata-rata)")
    rs = row.get("f_rs20", np.nan)
    if np.isfinite(rs):
        (pros if rs > 0 else cons).append(f"Relative strength 20H vs S&P 500 {_pct(rs)}")
    ss = row.get("sector_score", np.nan)
    if np.isfinite(ss):
        if ss >= 65:
            pros.append(f"Sektor {row['sector']} outperform (sector score {ss:.0f})")
        elif ss <= 35:
            cons.append(f"Sektor {row['sector']} lemah (sector score {ss:.0f})")
    if row["adx"] >= 25 and row["plus_di"] > row["minus_di"]:
        pros.append(f"Trend kuat: ADX {row['adx']:.0f}, +DI > -DI")
    if row["rsi14"] >= 75:
        cons.append(f"RSI14 {row['rsi14']:.0f} — overbought, rawan pullback")
    pros.append(f"Probability bullish {row['probability_bullish']:.0%} (bearish {row['probability_bearish']:.0%})")
    if row["atr_pct"] >= 0.05:
        cons.append(f"ATR tinggi ({row['atr_pct']:.1%} dari harga)")
    if row.get("tp_warning"):
        cons.append(row["tp_warning"])
    if row["market_regime"] in ("NEUTRAL", "BEAR", "STRONG_BEAR"):
        cons.append(f"Regime S&P 500 {row['market_regime']}")
    if row.get("prob_dispersion", 0) > 0.08:
        cons.append(f"Ketidaksepakatan model {row['prob_dispersion']:.2f}")
    ns = row.get("news_sent5", np.nan)
    if row.get("news_count5", 0) and isinstance(ns, float) and np.isfinite(ns) and abs(ns) > 0.2:
        (pros if ns > 0 else cons).append(f"Sentimen berita 5H {ns:+.2f} ({int(row['news_count5'])} berita)")
    for r in row.get("reject_reasons", []) or []:
        if r != "score_below_buy_min" or row.get("decision") != "BUY":
            cons.append("Filter: " + REJECT_TEXT.get(r, r))
    return pros, cons


def model_drivers(model, row_df: pd.DataFrame, candidates: list[str], top: int = 3) -> list[str]:
    """Δ probability_bullish bila fitur diganti median training (sensitivity lokal)."""
    med = model.meta.get("feature_medians", {})
    base = model.predict(row_df)["probability_bullish"].iloc[0]
    rows = []
    for f in candidates:
        if f not in med or f not in row_df:
            continue
        alt = row_df.copy()
        alt[f] = med[f]
        rows.append((f, base - model.predict(alt)["probability_bullish"].iloc[0]))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return [f"{feature_label(f)} ({f}): {d * 100:+.1f} poin persen ke P(bullish)" for f, d in rows[:top] if abs(d) >= 0.005]
