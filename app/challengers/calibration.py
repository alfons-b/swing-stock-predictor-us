"""Kalibrasi probabilitas challenger (Platt / sigmoid — sama dengan default model aktif `model.calibration: sigmoid`).

Dipasang pada prakiraan FOLD VALIDASI (model tahap seleksi; ARIMA-GARCH: parameter dari data ≤ akhir training validasi),
lalu diterapkan apa adanya pada periode test dan prakiraan harian. Pemetaan monoton → AUC & urutan top-k tidak berubah;
yang membaik adalah nilai probabilitas (ECE, log loss).
"""
from __future__ import annotations

import numpy as np

TARGETS = ("prob_bullish", "prob_bearish", "prob_up")
EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray, min_n: int = 200) -> dict | None:
    """P_cal = σ(a·logit(p) + b). None bila data terlalu sedikit atau satu kelas saja."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if len(p) < min_n or y.min() == y.max():
        return None
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n": int(len(p)), "base_rate": float(y.mean())}


def apply_platt(p: np.ndarray, params: dict | None) -> np.ndarray:
    if not params:
        return np.asarray(p, float)
    z = params["a"] * _logit(p) + params["b"]
    return 1.0 / (1.0 + np.exp(-z))


def fit_calibrators(fc: dict, y_ret: np.ndarray, bull_threshold: float, bear_threshold: float,
                    mask: np.ndarray | None = None) -> dict:
    """Satu kalibrator per target, dari prakiraan & label pada baris `mask` (fold validasi, tradeable, label ada)."""
    y_ret = np.asarray(y_ret, float)
    m = np.isfinite(y_ret) if mask is None else (np.asarray(mask, bool) & np.isfinite(y_ret))
    labels = {"prob_bullish": y_ret > bull_threshold, "prob_bearish": y_ret < bear_threshold, "prob_up": y_ret > 0}
    out = {}
    for t in TARGETS:
        cal = fit_platt(np.asarray(fc[t], float)[m], labels[t][m].astype(int))
        if cal:
            out[t] = cal
    return out


def calibrate_forecasts(fc: dict, calibrators: dict | None) -> dict:
    """Terapkan kalibrator; jaga P(bullish) + P(bearish) ≤ 1 dan hitung ulang P(netral)."""
    if not calibrators:
        return fc
    out = dict(fc)
    for t in TARGETS:
        if t in calibrators:
            out[t] = apply_platt(fc[t], calibrators[t])
    bull, bear = np.asarray(out["prob_bullish"], float), np.asarray(out["prob_bearish"], float)
    total = bull + bear
    scale = np.where(total > 0.999, 0.999 / np.maximum(total, 1e-12), 1.0)
    out["prob_bullish"], out["prob_bearish"] = bull * scale, bear * scale
    out["prob_neutral"] = np.clip(1 - out["prob_bullish"] - out["prob_bearish"], 0.0, 1.0)
    return out
