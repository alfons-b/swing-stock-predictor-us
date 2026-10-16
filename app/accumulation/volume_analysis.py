"""Analisis volume (per emiten, urut tanggal). Semua jendela memakai data s.d. hari itu saja (tanpa look-ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rvol(v: pd.Series, n: int = 20) -> pd.Series:
    """Relative volume: volume hari ini / rata-rata n hari SEBELUMNYA (hari ini tidak ikut pembagi)."""
    return v / v.shift(1).rolling(n, min_periods=n).mean().replace(0, np.nan)


def up_down_volume_ratio(c: pd.Series, v: pd.Series, n: int = 20) -> pd.Series:
    """Σ volume hari naik / Σ volume hari turun (n hari). Tidak ada hari turun → NaN (bukan tak hingga)."""
    d = c.diff()
    up = v.where(d > 0, 0.0).rolling(n, min_periods=n).sum()
    dn = v.where(d < 0, 0.0).rolling(n, min_periods=n).sum()
    return up / dn.replace(0, np.nan)


def volume_dry_up(v: pd.Series, short: int = 10, long: int = 60) -> pd.Series:
    """Rata-rata volume pendek / panjang. < 0,7 = volume mengering (tipikal fase base/akumulasi diam-diam)."""
    return v.rolling(short, min_periods=short).mean() / v.rolling(long, min_periods=long).mean().replace(0, np.nan)


def distribution_days(c: pd.Series, v: pd.Series, n: int = 20, drop: float = -0.01, vol_mult: float = 1.2) -> pd.Series:
    """Jumlah hari turun ≥ 1% dengan volume > 1,2 × rata-rata 50 hari sebelumnya, dalam n hari."""
    ret = c.pct_change()
    base = v.shift(1).rolling(50, min_periods=30).mean()
    flag = ((ret <= drop) & (v > vol_mult * base)).astype(float)
    return flag.rolling(n, min_periods=n).sum()


def accumulation_days(c: pd.Series, v: pd.Series, n: int = 20, rise: float = 0.01, vol_mult: float = 1.2) -> pd.Series:
    ret = c.pct_change()
    base = v.shift(1).rolling(50, min_periods=30).mean()
    flag = ((ret >= rise) & (v > vol_mult * base)).astype(float)
    return flag.rolling(n, min_periods=n).sum()
