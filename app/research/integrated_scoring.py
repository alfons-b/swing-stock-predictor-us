"""Skor terpisah + 9 ranking terintegrasi + keputusan VALUE (terpisah dari keputusan SWING).

Skor terpisah (0–100, None = data tidak tersedia — TIDAK pernah diisi nol):
  value_score, business_quality_score, value_trap_risk (LOW/MEDIUM/HIGH/UNKNOWN), foreign_flow_score,
  accumulation_score, distribution_risk_score, technical_setup_score, ml_probability_score, liquidity_score,
  market_regime_score
Turunan untuk ranking: value_risk_inverse (LOW 100 / MEDIUM 55 / HIGH 10 / UNKNOWN None),
  distribution_risk_inverse = 100 − distribution_risk_score.
Composite ranking = rata-rata berbobot skor yang TERSEDIA; bobot dinormalisasi ulang; bila bobot tersedia
< min_weight_coverage → composite None (emiten tidak diranking, bukan diberi nilai rendah).

Keputusan:
  swing_decision  BUY / WATCHLIST / WAIT / AVOID — dari pipeline swing (tidak diubah oleh modul valuasi)
  value_decision  VALUE_CANDIDATE / VALUE_WATCH / VALUE_TRAP_RISK / NOT_UNDERVALUED / INSUFFICIENT_DATA
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.scoring import REGIME_SCORE

SCORE_COLS = ["value_score", "business_quality_score", "value_trap_risk", "foreign_flow_score", "accumulation_score",
              "distribution_risk_score", "technical_setup_score", "ml_probability_score", "liquidity_score",
              "market_regime_score"]
TRAP_INV = {"LOW": 100.0, "MEDIUM": 55.0, "HIGH": 10.0}
RANKINGS = ["best_value", "quality_value", "accumulation", "foreign_buying", "swing_setup", "value_accumulation",
            "value_swing", "momentum_flow", "integrated"]


def assemble(sig: pd.DataFrame, val: pd.DataFrame | None, flow_last: pd.DataFrame | None,
             acc: pd.DataFrame | None) -> pd.DataFrame:
    """Satu baris per emiten (tanggal scan). sig: keluaran scan (generate_signals)."""
    t = pd.DataFrame({"ticker": sig["ticker"].values})
    for c in ("name", "sector", "close", "decision", "setup_type", "market_regime", "confidence"):
        if c in sig:
            t[c] = sig[c].values
    t = t.rename(columns={"decision": "swing_decision", "confidence": "swing_confidence"})
    t["technical_setup_score"] = pd.to_numeric(sig.get("technical_score"), errors="coerce").values
    t["ml_probability_score"] = pd.to_numeric(sig.get("ml_score"), errors="coerce").values
    t["liquidity_score"] = pd.to_numeric(sig.get("liquidity_score"), errors="coerce").values
    t["market_regime_score"] = pd.to_numeric(sig["market_regime"].astype(object).map(REGIME_SCORE), errors="coerce").values \
        if "market_regime" in sig else np.nan
    if val is not None and len(val):
        v = val.reindex(columns=["ticker", "value_score", "quality_score", "value_trap_risk", "valuation_status",
                                 "valuation_confidence", "margin_of_safety", "margin_of_safety_conservative",
                                 "fair_value_low", "fair_value_base", "fair_value_high", "sector_type"]).rename(columns={"quality_score": "business_quality_score"})
        t = t.merge(v, on="ticker", how="left")
    else:
        t["valuation_status"] = "INSUFFICIENT_DATA"
    if flow_last is not None and len(flow_last):
        t = t.merge(flow_last[["ticker", "foreign_flow_score", "foreign_flow_status", "foreign_flow_confidence"]],
                    on="ticker", how="left")
    if acc is not None and len(acc):
        a = acc[["ticker", "accumulation_score", "distribution_risk", "accumulation_status", "accumulation_stage",
                 "accumulation_confidence"]].rename(columns={"distribution_risk": "distribution_risk_score"})
        t = t.merge(a, on="ticker", how="left")
    for c in SCORE_COLS:
        if c not in t:
            t[c] = np.nan if c != "value_trap_risk" else "UNKNOWN"
    t["value_trap_risk"] = t["value_trap_risk"].fillna("UNKNOWN")
    t["valuation_status"] = t.get("valuation_status", pd.Series(index=t.index, dtype=object)).fillna("INSUFFICIENT_DATA")
    t["foreign_flow_status"] = t.get("foreign_flow_status", pd.Series(index=t.index, dtype=object)).fillna(
        "FOREIGN_FLOW_UNAVAILABLE")
    t["value_risk_inverse"] = t["value_trap_risk"].map(TRAP_INV)
    t["distribution_risk_inverse"] = 100 - pd.to_numeric(t["distribution_risk_score"], errors="coerce")
    t["value_decision"] = [value_decision(r) for r in t.to_dict("records")]
    return t


def value_decision(r: dict) -> str:
    st = r.get("valuation_status")
    if st in (None, "INSUFFICIENT_DATA") or (isinstance(st, float) and np.isnan(st)):
        return "INSUFFICIENT_DATA"
    if st not in ("DEEP_VALUE", "UNDERVALUED"):
        return "NOT_UNDERVALUED"
    if r.get("value_trap_risk") == "HIGH":
        return "VALUE_TRAP_RISK"
    q = r.get("business_quality_score")
    q_ok = q is not None and not (isinstance(q, float) and np.isnan(q)) and q >= 50
    if r.get("value_trap_risk") == "LOW" and q_ok and r.get("valuation_confidence") in ("HIGH", "MEDIUM"):
        return "VALUE_CANDIDATE"
    return "VALUE_WATCH"


def composite(t: pd.DataFrame, weights: dict, min_coverage: float) -> pd.Series:
    num = pd.Series(0.0, index=t.index)
    den = pd.Series(0.0, index=t.index)
    for k, w in weights.items():
        s = pd.to_numeric(t[k], errors="coerce") if k in t else pd.Series(np.nan, index=t.index)
        num += s.fillna(0) * w
        den += s.notna() * w
    total = sum(weights.values()) or 1.0
    return (num / den.replace(0, np.nan)).where(den / total >= min_coverage)


def rank_all(t: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rc = cfg.get("research_scoring", {}) or {}
    specs = rc.get("rankings", {}) or {}
    cov = float(rc.get("min_weight_coverage", 0.6))
    t = t.copy()
    for name in RANKINGS:
        w = specs.get(name)
        if not w:
            continue
        s = composite(t, w, cov)
        if name in ("best_value", "quality_value", "value_accumulation", "value_swing"):
            s = s.where(t["value_score"].notna())                       # ranking nilai butuh valuasi yang valid
            s = s.where(t["value_trap_risk"] != "HIGH")                 # value trap tinggi tidak masuk ranking nilai
        if name in ("accumulation", "value_accumulation", "momentum_flow"):
            s = s.where(t["accumulation_score"].notna())
        if name == "foreign_buying":
            s = s.where(t["foreign_flow_score"].notna())
        t[f"score_{name}"] = s.round(1)
        t[f"rank_{name}"] = s.rank(ascending=False, method="first")
    return t


def top(t: pd.DataFrame, name: str, n: int = 10) -> pd.DataFrame:
    col = f"rank_{name}"
    if col not in t:
        return t.head(0)
    return t[t[col].notna()].sort_values(col).head(n)
