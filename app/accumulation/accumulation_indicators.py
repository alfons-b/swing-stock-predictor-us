"""Indikator akumulasi/distribusi per emiten: OBV, ADL, CMF, MFI, RVOL, CLV, VWAP, divergensi, base, breakout.

Rumus inti ada di app/features/indicators.py (dipakai bersama, tidak diduplikasi). Kolom keluaran berprefiks `acc_`
(bukan `f_`) sehingga model ML aktif tidak terpengaruh; bisa dievaluasi sebagai fitur di versi model berikutnya.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.accumulation import price_volume_analysis as pva
from app.accumulation import volume_analysis as va
from app.features import indicators as ind


def compute_one(g: pd.DataFrame, window: int = 20, long_window: int = 60) -> pd.DataFrame:
    g = g.sort_values("date")
    h, l, c, v = (g[k].astype(float) for k in ("high", "low", "close", "volume"))
    value = g["value"].astype(float) if "value" in g else None
    out = pd.DataFrame({"ticker": g["ticker"].values, "date": g["date"].values}, index=g.index)
    obv, adl = ind.obv(c, v), ind.adl(h, l, c, v)
    clv = ind.clv(h, l, c)
    out["acc_obv_trend_20"] = pva.flow_trend(obv, v, window)
    out["acc_obv_trend_60"] = pva.flow_trend(obv, v, long_window)
    out["acc_adl_trend_20"] = pva.flow_trend(adl, v, window)
    out["acc_cmf_20"] = ind.cmf(h, l, c, v, window)
    out["acc_mfi_14"] = ind.mfi(h, l, c, v, 14)
    out["acc_rvol_20"] = va.rvol(v, window)
    out["acc_clv"] = clv
    out["acc_clv_avg_20"] = clv.rolling(window, min_periods=window).mean()
    vwap = ind.rolling_vwap(h, l, c, v, window, value)
    out["acc_vwap_20"] = vwap
    out["acc_price_vs_vwap"] = c / vwap - 1
    out["acc_up_down_vol_20"] = va.up_down_volume_ratio(c, v, window)
    out["acc_vol_dry_up"] = va.volume_dry_up(v, 10, long_window)
    out["acc_distribution_days_20"] = va.distribution_days(c, v, window)
    out["acc_accumulation_days_20"] = va.accumulation_days(c, v, window)
    out["acc_base_range_20"] = pva.base_range(h, l, c, window)
    out["acc_breakout"] = pva.breakout(h, c, out["acc_rvol_20"], clv, window)
    out["acc_breakout_recent"] = out["acc_breakout"].rolling(3, min_periods=1).max()
    out["acc_divergence_20"] = pva.divergence(c, out["acc_obv_trend_20"], window)
    out["acc_ret_20"] = c / c.shift(window) - 1
    near_high = c >= 0.97 * h.rolling(window, min_periods=window).max()
    churn = ((out["acc_rvol_20"] > 1.5) & (c.pct_change().abs() < 0.005) & near_high).astype(float)
    out["acc_churn_10"] = churn.rolling(10, min_periods=10).sum()
    out["acc_avg_value_20"] = (value if value is not None else c * v).rolling(window, min_periods=window).mean()
    out["acc_history_days"] = np.arange(1, len(g) + 1)
    return out


def compute_indicators(prices: pd.DataFrame, window: int = 20, long_window: int = 60) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    parts = [compute_one(g, window, long_window) for _, g in prices.groupby("ticker", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out
