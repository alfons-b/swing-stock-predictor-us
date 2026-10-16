"""Fitur per-saham: trend, momentum, volatilitas, volume, struktur harga, return, gap.

Fungsi `compute_stock_features` menerima SATU ticker (urut tanggal) dan mengembalikan
kolom indikator mentah (untuk strategi) + kolom `f_*` (ternormalisasi, untuk ML).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features import indicators as ind


def compute_stock_features(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
    val = g["value"]
    out = {}

    # ---------- Trend
    for n in (5, 10, 20, 50, 100, 200):
        out[f"sma{n}"] = ind.sma(c, n)
        out[f"f_dist_sma{n}"] = c / out[f"sma{n}"] - 1
    for n in (9, 20, 50, 200):
        out[f"ema{n}"] = ind.ema(c, n)
    for n in (9, 20, 50):
        out[f"f_dist_ema{n}"] = c / out[f"ema{n}"] - 1
    out["f_ema20_slope5"] = ind.slope(out["ema20"], 5)
    out["f_ema20_over_ema50"] = out["ema20"] / out["ema50"] - 1
    out["f_sma50_over_sma200"] = out["sma50"] / out["sma200"] - 1
    out["f_ema_stack"] = ((c > out["ema20"]).astype(float) + (out["ema20"] > out["ema50"]).astype(float)
                          + (out["ema50"] > out["sma200"]).astype(float)).where(out["sma200"].notna())

    # ---------- Momentum
    for n in (7, 14, 21):
        out[f"rsi{n}"] = ind.rsi(c, n)
        out[f"f_rsi{n}"] = out[f"rsi{n}"]
    out["f_roc10"] = c / c.shift(10) - 1
    out["f_mom10_atr"] = (c - c.shift(10))  # dinormalisasi ATR di bawah
    out["stoch_k"], out["stoch_d"] = ind.stochastic(h, l, c)
    out["f_stoch_k"], out["f_stoch_d"] = out["stoch_k"], out["stoch_d"]
    out["f_williams_r"] = ind.williams_r(h, l, c)

    # ---------- Trend strength
    out["adx"], out["plus_di"], out["minus_di"] = ind.adx(h, l, c)
    out["f_adx"], out["f_di_diff"] = out["adx"], out["plus_di"] - out["minus_di"]

    # ---------- MACD (dinormalisasi harga)
    m_line, m_sig, m_hist = ind.macd(c)
    out["macd_hist"] = m_hist
    out["f_macd"] = m_line / c
    out["f_macd_signal"] = m_sig / c
    out["f_macd_hist"] = m_hist / c
    out["f_macd_hist_slope"] = (m_hist - m_hist.shift(3)) / c

    # ---------- Volatility
    out["atr14"] = ind.atr(h, l, c, 14)
    out["atr_pct"] = out["atr14"] / c
    out["f_atr_pct"] = out["atr_pct"]
    # Pembagi ATR diberi batas bawah 0,01% harga: saat suspensi panjang (harga beku) ATR Wilder menyusut
    # eksponensial ke ~1e-70, sehingga rasio terhadap ATR meledak (>1e38, melebihi float32) begitu saham aktif lagi.
    atr_safe = np.maximum(out["atr14"], c * 1e-4)
    atr_pct_safe = atr_safe / c
    out["f_mom10_atr"] = out["f_mom10_atr"] / atr_safe
    ret1 = c.pct_change()
    out["vol20"] = ret1.rolling(20, min_periods=20).std()
    out["f_vol20"] = out["vol20"]
    out["f_vol_ratio_20_60"] = out["vol20"] / ret1.rolling(60, min_periods=60).std()
    out["bb_width"] = ind.bollinger_width(c)
    out["f_bb_width"] = out["bb_width"]
    out["bb_width_pct"] = ind.rolling_pct_rank(out["bb_width"], 120)
    out["f_bb_width_pct"] = out["bb_width_pct"]
    out["f_range_pct"] = (h - l) / c
    out["f_atr_contraction"] = atr_pct_safe / atr_pct_safe.rolling(60, min_periods=30).median()

    # ---------- Volume
    out["vol_sma20"] = ind.sma(v, 20)
    out["volume_ratio"] = v / out["vol_sma20"].replace(0, np.nan)
    out["f_volume_ratio"] = out["volume_ratio"]
    out["f_volume_spike"] = (out["volume_ratio"] > 2.0).astype(float)
    obv = ind.obv(c, v)
    out["f_obv_slope"] = (obv - obv.shift(10)) / (out["vol_sma20"] * 10).replace(0, np.nan)
    out["f_volume_mom"] = ind.sma(v, 5) / out["vol_sma20"].replace(0, np.nan) - 1
    out["avg_value20"] = ind.sma(val, 20)
    out["median_value20"] = val.rolling(20, min_periods=20).median()
    out["avg_volume20"] = out["vol_sma20"]
    out["active_days20"] = (v > 0).astype(float).rolling(20, min_periods=1).sum()
    out["f_value_mom"] = ind.sma(val, 5) / out["avg_value20"].replace(0, np.nan) - 1
    out["f_log_value20"] = np.log1p(out["avg_value20"])
    out["f_close_pos"] = (c - l) / (h - l).replace(0, np.nan)

    # ---------- Price structure (resistance/support = hari SEBELUMNYA → close hari ini bisa breakout)
    for n in (20, 50):
        out[f"res{n}"] = h.shift(1).rolling(n, min_periods=n).max()
        out[f"sup{n}"] = l.shift(1).rolling(n, min_periods=n).min()
        out[f"f_dist_high{n}"] = c / h.rolling(n, min_periods=n).max() - 1
        out[f"f_dist_low{n}"] = c / l.rolling(n, min_periods=n).min() - 1
    out["res120"] = h.shift(1).rolling(120, min_periods=60).max()
    out["f_res_dist"] = out["res20"] / c - 1
    out["f_sup_dist"] = c / out["sup20"] - 1
    out["breakout20"] = (c > out["res20"]).astype(float)
    out["f_breakout20"] = out["breakout20"]
    out["f_breakout50"] = (c > out["res50"]).astype(float)
    hh10 = h.rolling(10, min_periods=10).max()
    ll10 = l.rolling(10, min_periods=10).min()
    out["higher_high"] = (hh10 > hh10.shift(10)).astype(float)
    out["higher_low"] = (ll10 > ll10.shift(10)).astype(float)
    out["lower_high"] = (hh10 < hh10.shift(10)).astype(float)
    out["lower_low"] = (ll10 < ll10.shift(10)).astype(float)
    out["f_structure"] = out["higher_high"] + out["higher_low"] - out["lower_high"] - out["lower_low"]
    out["swing_low10"] = l.rolling(10, min_periods=10).min()
    base_n = cfg.get("setups", {}).get("base_lookback", 40)
    out["base_high"] = h.shift(1).rolling(base_n, min_periods=base_n).max()
    out["base_low"] = l.shift(1).rolling(base_n, min_periods=base_n).min()
    out["base_range"] = out["base_high"] / out["base_low"] - 1
    out["f_base_range"] = out["base_range"]

    # ---------- Returns
    for n in (1, 3, 5, 10, 20, 60):
        out[f"ret{n}"] = c / c.shift(n) - 1
        out[f"f_ret{n}"] = out[f"ret{n}"]

    # ---------- Gap
    pc = c.shift(1)
    gap = o / pc - 1
    out["f_gap"] = gap
    gap_up = gap > 0.01
    filled = gap_up & (l <= pc)
    out["f_gap_continuation"] = (gap_up & (c > o)).astype(float)
    out["f_gap_fill_rate"] = filled.astype(float).rolling(120, min_periods=20).sum() / \
        gap_up.astype(float).rolling(120, min_periods=20).sum().replace(0, np.nan)

    feats = pd.DataFrame(out, index=g.index)
    res = pd.concat([g, feats], axis=1)
    return res
