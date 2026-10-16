"""Skor foreign flow 0–100 dan klasifikasi.

Skor = rata-rata berbobot (komponen yang tersedia saja):
  net ratio 20H (45%), net ratio 5H (25%), porsi hari net beli 20H (20%), net ratio 60H (10%)
  net ratio dipetakan linear −15%..+15% volume → 0..100.
Status:
  FOREIGN_FLOW_UNAVAILABLE  tidak ada data foreign flow sama sekali dalam 20 hari bursa terakhir
  INSUFFICIENT_DATA   cakupan data 20H < min_coverage (data sebagian; tidak ada data ≠ netral)
  STRONG_NET_BUYING   skor ≥ 75, net 20H > 0, hari net beli ≥ 60%
  NET_BUYING          skor ≥ 60 dan net 20H > 0
  STRONG_NET_SELLING  skor ≤ 25, net 20H < 0, hari net beli ≤ 40%
  NET_SELLING         skor ≤ 40 dan net 20H < 0
  NEUTRAL             lainnya
Partisipasi asing < 2% volume → `low_participation` (sinyal kurang bermakna; confidence LOW).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUSES = ["STRONG_NET_BUYING", "NET_BUYING", "NEUTRAL", "NET_SELLING", "STRONG_NET_SELLING", "INSUFFICIENT_DATA",
            "FOREIGN_FLOW_UNAVAILABLE"]


def _lin(x, lo=-0.15, hi=0.15):
    return ((x - lo) / (hi - lo)).clip(0, 1) * 100


def score_flows(feat: pd.DataFrame, min_coverage: float = 0.8) -> pd.DataFrame:
    out = feat[["ticker", "date"]].copy()
    if "flow_net_ratio_20" not in feat:
        out["foreign_flow_score"] = np.nan
        out["foreign_flow_status"] = "FOREIGN_FLOW_UNAVAILABLE"
        out["foreign_flow_confidence"] = None
        return out
    parts = {"n20": (_lin(feat["flow_net_ratio_20"]), 0.45), "n5": (_lin(feat.get("flow_net_ratio_5")), 0.25),
             "days": (feat.get("flow_buy_days_ratio_20") * 100, 0.20), "n60": (_lin(feat.get("flow_net_ratio_60")), 0.10)}
    num = sum(v.fillna(0) * w for v, w in parts.values())
    den = sum(v.notna() * w for v, w in parts.values())
    score = (num / den.replace(0, np.nan))
    cov = feat.get("flow_coverage_20", pd.Series(0.0, index=feat.index)).fillna(0)
    ok = cov >= min_coverage
    score = score.where(ok)
    n20 = feat["flow_net_ratio_20"]
    days = feat.get("flow_buy_days_ratio_20", pd.Series(np.nan, index=feat.index))
    status = np.select(
        [cov <= 0, ~ok, (score >= 75) & (n20 > 0) & (days >= 0.6), (score >= 60) & (n20 > 0),
         (score <= 25) & (n20 < 0) & (days <= 0.4), (score <= 40) & (n20 < 0)],
        ["FOREIGN_FLOW_UNAVAILABLE", "INSUFFICIENT_DATA", "STRONG_NET_BUYING", "NET_BUYING", "STRONG_NET_SELLING", "NET_SELLING"], "NEUTRAL")
    part = feat.get("flow_participation_20", pd.Series(np.nan, index=feat.index))
    out["foreign_flow_score"] = score
    out["foreign_flow_status"] = status
    out["low_participation"] = (part < 0.02) & ok
    out["foreign_flow_confidence"] = np.where(~ok, None, np.where(out["low_participation"] | (cov < 0.95), "LOW",
                                                                   np.where(feat.get("flow_coverage_60", cov) >= 0.9,
                                                                            "HIGH", "MEDIUM")))
    return out
