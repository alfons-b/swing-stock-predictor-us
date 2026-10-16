"""Hubungan harga-volume: posisi terhadap VWAP, base/kontraksi range, breakout, divergensi."""
from __future__ import annotations

import numpy as np
import pandas as pd


def flow_trend(cum: pd.Series, v: pd.Series, n: int = 20) -> pd.Series:
    """Perubahan indikator kumulatif (OBV/ADL) n hari dinormalisasi Σ volume n hari → kira-kira −1..1."""
    return (cum - cum.shift(n)) / v.rolling(n, min_periods=n).sum().replace(0, np.nan)


def base_range(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 20) -> pd.Series:
    """(High n hari − Low n hari) / close: < 15% = konsolidasi sempit."""
    return (h.rolling(n, min_periods=n).max() - l.rolling(n, min_periods=n).min()) / c


def breakout(h: pd.Series, c: pd.Series, rv: pd.Series, clv_: pd.Series, n: int = 20, min_rvol: float = 1.5,
             min_clv: float = 0.3) -> pd.Series:
    """Close menembus high n hari SEBELUMNYA dengan volume relatif ≥ min_rvol dan close di bagian atas range."""
    prior_high = h.shift(1).rolling(n, min_periods=n).max()
    return ((c > prior_high) & (rv >= min_rvol) & (clv_ >= min_clv)).astype(float).where(prior_high.notna())


def divergence(c: pd.Series, flow_norm: pd.Series, n: int = 20, price_th: float = 0.03, flow_th: float = 0.05) -> pd.Series:
    """+1 bullish (harga turun, aliran volume naik), −1 bearish (harga naik, aliran turun), 0 tidak ada."""
    pc = c / c.shift(n) - 1
    out = np.where((pc < -price_th) & (flow_norm > flow_th), 1.0, np.where((pc > price_th) & (flow_norm < -flow_th), -1.0, 0.0))
    return pd.Series(out, index=c.index).where(pc.notna() & flow_norm.notna())
