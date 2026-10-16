"""Skor akumulasi 0–100, status, tahap, dan confidence.

Prinsip kejujuran sinyal:
- Ini bukti PERILAKU harga-volume (+ foreign flow bila ada), bukan bukti identitas pembeli ("bandar").
- STRONG_* hanya bila bukti kunci lengkap: skor tinggi + CMF & OBV searah + foreign flow TERSEDIA dan searah.
  Tanpa foreign flow (atau broker summary) status maksimum MODERATE_* dan dicatat di evidence.
- Histori < 60 hari bursa → INSUFFICIENT_DATA.

Status : STRONG_ACCUMULATION_SIGNAL | MODERATE_ACCUMULATION_SIGNAL | NEUTRAL | MODERATE_DISTRIBUTION_SIGNAL |
         STRONG_DISTRIBUTION_SIGNAL | INSUFFICIENT_DATA
Tahap  : EARLY_ACCUMULATION_WATCHLIST | BREAKOUT_CONFIRMED | DISTRIBUTION_WARNING | NO_CLEAR_SIGNAL
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.accumulation.distribution_detector import distribution_risk

WEIGHTS = {"cmf": 0.15, "obv_trend": 0.15, "adl_trend": 0.10, "mfi": 0.10, "up_down_volume": 0.15, "close_location": 0.10,
           "vwap_position": 0.05, "divergence": 0.05, "foreign_flow": 0.20}
MIN_HISTORY = 60


def _lin(x, lo, hi):
    return ((x - lo) / (hi - lo)).clip(0, 1) * 100


def components(ind: pd.DataFrame, flow_score: pd.Series | None) -> pd.DataFrame:
    c = pd.DataFrame(index=ind.index)
    c["cmf"] = _lin(ind["acc_cmf_20"], -0.2, 0.2)
    c["obv_trend"] = _lin(ind["acc_obv_trend_20"], -0.3, 0.3)
    c["adl_trend"] = _lin(ind["acc_adl_trend_20"], -0.3, 0.3)
    c["mfi"] = _lin(ind["acc_mfi_14"], 20, 80)
    c["up_down_volume"] = _lin(np.log(ind["acc_up_down_vol_20"].clip(lower=1e-6)), np.log(0.5), np.log(2.0))
    c["close_location"] = _lin(ind["acc_clv_avg_20"], -0.4, 0.4)
    c["vwap_position"] = _lin(ind["acc_price_vs_vwap"], -0.05, 0.05)
    div = ind["acc_divergence_20"]
    c["divergence"] = np.where(div > 0, 100.0, np.where(div < 0, 0.0, np.nan))
    c["foreign_flow"] = flow_score if flow_score is not None else np.nan
    return c


def score(ind: pd.DataFrame, flow: pd.DataFrame | None = None, thresholds: dict | None = None,
          min_value: float = 1e9) -> pd.DataFrame:
    """ind: keluaran compute_indicators; flow: kolom ticker,date,foreign_flow_score,foreign_flow_status (opsional)."""
    th = {"strong": 75, "moderate": 60, "weak_distribution": 40, "strong_distribution": 25, **(thresholds or {})}
    d = ind.copy()
    if flow is not None and len(flow):
        d = d.merge(flow[["ticker", "date", "foreign_flow_score", "foreign_flow_status"]], on=["ticker", "date"], how="left")
    else:
        d["foreign_flow_score"], d["foreign_flow_status"] = np.nan, "FOREIGN_FLOW_UNAVAILABLE"
    d["foreign_flow_status"] = d["foreign_flow_status"].fillna("FOREIGN_FLOW_UNAVAILABLE")
    fs = d["foreign_flow_score"]
    comp = components(d, fs)
    w = pd.Series(WEIGHTS)
    num = (comp.fillna(0) * w).sum(axis=1)
    den = (comp.notna() * w).sum(axis=1)
    s = (num / den.replace(0, np.nan)).where(den >= 0.5)
    enough = d["acc_history_days"] >= MIN_HISTORY
    s = s.where(enough)
    risk, _ = distribution_risk(d, fs)
    risk = risk.where(enough)
    has_flow = fs.notna()
    key_acc = (d["acc_cmf_20"] > 0.05) & (d["acc_obv_trend_20"] > 0)
    key_dist = (d["acc_cmf_20"] < -0.05) & (d["acc_obv_trend_20"] < 0)
    status = np.select(
        [~enough | s.isna(),
         (s >= th["strong"]) & key_acc & has_flow & (fs >= 60),
         s >= th["moderate"],
         (s <= th["strong_distribution"]) & key_dist & has_flow & (fs <= 40),
         s <= th["weak_distribution"]],
        ["INSUFFICIENT_DATA", "STRONG_ACCUMULATION_SIGNAL", "MODERATE_ACCUMULATION_SIGNAL", "STRONG_DISTRIBUTION_SIGNAL",
         "MODERATE_DISTRIBUTION_SIGNAL"], "NEUTRAL")
    in_base = (d["acc_base_range_20"] <= 0.15) | (d["acc_vol_dry_up"] < 0.8)
    stage = np.select(
        [~enough | s.isna(),
         (risk >= 60) | pd.Series(status, index=d.index).str.contains("DISTRIBUTION"),
         (d["acc_breakout_recent"] >= 1) & (s >= 50),
         (s >= th["moderate"]) & in_base & (d["acc_breakout_recent"] < 1)],
        ["NO_CLEAR_SIGNAL", "DISTRIBUTION_WARNING", "BREAKOUT_CONFIRMED", "EARLY_ACCUMULATION_WATCHLIST"], "NO_CLEAR_SIGNAL")
    illiquid = d["acc_avg_value_20"] < min_value
    conf = np.where(~enough, None, np.where(illiquid, "LOW", np.where(has_flow, "HIGH", "MEDIUM")))
    out = d[["ticker", "date"]].copy()
    out["accumulation_score"] = s
    out["distribution_risk"] = risk
    out["accumulation_status"] = status
    out["accumulation_stage"] = stage
    out["accumulation_confidence"] = conf
    out["foreign_flow_status"] = d["foreign_flow_status"]
    out["_components"] = comp.round(1).to_dict("records")
    out["_strong_capped"] = (s >= th["strong"]) & key_acc & ~has_flow
    out["_illiquid"] = illiquid
    return out


def evidence(row: dict, ind_row: dict) -> list[str]:
    """Kalimat bukti yang dapat diverifikasi untuk satu emiten-hari."""
    e = []
    g = lambda k: ind_row.get(k)  # noqa: E731
    if g("acc_cmf_20") is not None and np.isfinite(g("acc_cmf_20")):
        e.append(f"CMF20 {g('acc_cmf_20'):+.2f}")
    if g("acc_obv_trend_20") is not None and np.isfinite(g("acc_obv_trend_20")):
        e.append(f"OBV 20H {g('acc_obv_trend_20'):+.2f}× volume")
    if g("acc_up_down_vol_20") is not None and np.isfinite(g("acc_up_down_vol_20")):
        e.append(f"volume naik/turun {g('acc_up_down_vol_20'):.2f}")
    if g("acc_rvol_20") is not None and np.isfinite(g("acc_rvol_20")):
        e.append(f"RVOL {g('acc_rvol_20'):.2f}")
    if (g("acc_divergence_20") or 0) > 0:
        e.append("divergensi bullish (harga turun, OBV naik)")
    if (g("acc_divergence_20") or 0) < 0:
        e.append("divergensi bearish (harga naik, OBV turun)")
    if (g("acc_distribution_days_20") or 0) >= 4:
        e.append(f"{g('acc_distribution_days_20'):.0f} distribution days dalam 20H")
    if row.get("_strong_capped"):
        e.append("skor setara STRONG tetapi short flow tidak tersedia → dibatasi MODERATE")
    if row.get("_illiquid"):
        e.append("likuiditas rendah — pola volume mudah dipengaruhi transaksi kecil")
    if row.get("foreign_flow_status") == "FOREIGN_FLOW_UNAVAILABLE":
        e.append("FOREIGN_FLOW_UNAVAILABLE")
    return e
