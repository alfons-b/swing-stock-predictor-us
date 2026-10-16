"""Risiko distribusi 0–100 (tinggi = tanda distribusi/penjualan terselubung).

Komponen (hanya yang tersedia, dirata-rata):
  distribution days 20H (0 → 0, ≥ 6 → 100) · divergensi bearish (100) / bullish (0) · churning dekat puncak
  (0 → 0, ≥ 3 → 100) · CLV rata-rata negatif · CMF negatif saat harga di atas VWAP · net jual asing (100 − skor flow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _lin(x, lo, hi):
    return ((x - lo) / (hi - lo)).clip(0, 1) * 100


def distribution_risk(ind: pd.DataFrame, flow_score: pd.Series | None = None) -> tuple[pd.Series, pd.DataFrame]:
    comp = pd.DataFrame(index=ind.index)
    comp["distribution_days"] = _lin(ind["acc_distribution_days_20"], 0, 6)
    div = ind["acc_divergence_20"]
    comp["bearish_divergence"] = np.where(div < 0, 100.0, np.where(div > 0, 0.0, 30.0))
    comp.loc[div.isna(), "bearish_divergence"] = np.nan
    comp["churning"] = _lin(ind["acc_churn_10"], 0, 3)
    comp["weak_closes"] = _lin(-ind["acc_clv_avg_20"], -0.1, 0.4)
    hidden = (ind["acc_price_vs_vwap"] > 0) & (ind["acc_cmf_20"] < 0)
    comp["rising_price_outflow"] = np.where(hidden, _lin(-ind["acc_cmf_20"], 0, 0.2), 0.0)
    comp.loc[ind["acc_cmf_20"].isna(), "rising_price_outflow"] = np.nan
    if flow_score is not None:
        comp["foreign_selling"] = 100 - flow_score
    return comp.mean(axis=1, skipna=True).where(comp.notna().sum(axis=1) >= 3), comp
