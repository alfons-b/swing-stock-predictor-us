"""Distribusi prakiraan → besaran yang sama untuk semua model challenger (numpy saja).

Semua model memprakirakan distribusi LOG return kumulatif H hari (H = labels.SWING_HORIZON):
- jaringan saraf (LSTM/GRU/TFT): kuantil langsung (pinball loss);
- ARIMA-GARCH: parametrik (μ, σ) → kuantil normal.
Dari kuantil diturunkan: P(naik), P(bullish) = P(return > bull_threshold), P(bearish), ekspektasi return, σ, q10/q50/q90.
Ambang bullish/bearish adalah RETURN SEDERHANA (sama dengan label y_class) dan dikonversi ke log di sini.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

P_MIN, P_MAX = 1e-4, 1 - 1e-4


def normal_quantiles(mu: np.ndarray, sigma: np.ndarray, levels) -> np.ndarray:
    z = norm.ppf(np.asarray(levels, dtype=float))
    return np.asarray(mu, float)[:, None] + np.asarray(sigma, float)[:, None] * z[None, :]


def sigma_from_quantiles(q: np.ndarray, levels) -> np.ndarray:
    """σ dari kuantil terluar (normal-equivalent). Selalu > 0."""
    lv = np.asarray(levels, float)
    z = norm.ppf(lv[-1]) - norm.ppf(lv[0])
    return np.maximum((q[:, -1] - q[:, 0]) / z, 1e-6)


def quantile_cdf(q: np.ndarray, levels, x) -> np.ndarray:
    """P(X ≤ x) dari kuantil monoton per baris. Di dalam rentang kuantil: interpolasi linear; di luar: ekor normal
    (σ dari kuantil terluar) yang dibatasi agar tetap monoton terhadap level terluar."""
    q = np.asarray(q, float)
    lv = np.asarray(levels, float)
    n = q.shape[0]
    x = np.broadcast_to(np.asarray(x, float), (n,)).copy()
    p = np.full(n, np.nan)
    for j in range(len(lv) - 1):
        lo, hi = q[:, j], q[:, j + 1]
        inside = np.isnan(p) & (x >= lo) & (x <= hi)
        width = hi - lo
        frac = np.where(width > 0, (x - lo) / np.where(width > 0, width, 1.0), 1.0)
        p = np.where(inside, lv[j] + frac * (lv[j + 1] - lv[j]), p)
    sig = sigma_from_quantiles(q, lv)
    med = q[:, int(np.argmin(np.abs(lv - 0.5)))]
    tail = norm.cdf((x - med) / sig)
    below, above = np.isnan(p) & (x < q[:, 0]), np.isnan(p) & (x > q[:, -1])
    p = np.where(below, np.minimum(tail, lv[0]), p)
    p = np.where(above, np.maximum(tail, lv[-1]), p)
    return np.clip(p, P_MIN, P_MAX)


def summarize_quantiles(q_log: np.ndarray, levels, bull_threshold: float, bear_threshold: float) -> dict:
    """Kuantil log return → kolom standar (return SEDERHANA kecuali `vol` yang σ log return H hari)."""
    q_log = np.sort(np.asarray(q_log, float), axis=1)          # jaga monoton (no crossing)
    lv = np.asarray(levels, float)
    sig = sigma_from_quantiles(q_log, lv)
    med = q_log[:, int(np.argmin(np.abs(lv - 0.5)))]
    p_le0 = quantile_cdf(q_log, lv, 0.0)
    p_le_bull = quantile_cdf(q_log, lv, np.log1p(bull_threshold))
    p_le_bear = quantile_cdf(q_log, lv, np.log1p(bear_threshold))
    prob_bull = 1 - p_le_bull
    prob_bear = p_le_bear
    out = {"exp_return": np.expm1(med + 0.5 * sig ** 2), "prob_up": 1 - p_le0, "prob_bullish": prob_bull,
           "prob_bearish": prob_bear, "prob_neutral": np.clip(1 - prob_bull - prob_bear, 0.0, 1.0), "vol": sig}
    for lvl, name in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
        j = int(np.argmin(np.abs(lv - lvl)))
        out[name] = np.expm1(q_log[:, j])
    return out
