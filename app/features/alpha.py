"""Fitur alpha cross-sectional (versi fitur f2) — sinyal dengan bukti empiris di saham AS, bebas look-ahead.

Mengapa (audit 16 Okt 2026): semua fitur f1 adalah indikator teknikal per saham dalam skala mentah. Model belajar
"saham mana yang volatil" dan "pasar sedang naik", bukan "saham mana yang akan mengungguli saham lain". Fitur di sini
menambahkan sinyal yang diteliti luas dan menormalisasinya per tanggal (rank 0..1) agar stabil antar regime:

  Momentum 12-1, 6-1 ........ Jegadeesh & Titman (1993): pemenang 3–12 bulan terus unggul; bulan terakhir dilewati
                               karena ada efek pembalikan jangka pendek.
  Pembalikan 1 minggu/1 bulan  Jegadeesh (1990), Lehmann (1990): return 1–4 minggu cenderung berbalik; lebih kuat
                               bila dihitung dari return RESIDUAL (dikurangi beta × pasar) — Blitz dkk. (2013).
  Jarak ke high 52 minggu ... George & Hwang (2004).
  MAX (return harian maks 21H)  Bali, Cakici & Whitelaw (2011): saham "lotre" cenderung underperform.
  Volatilitas idiosinkratik .. Ang dkk. (2006).
  Beta 60H .................. dipakai untuk residualisasi dan sebagai fitur risiko.

Semua jendela hanya memakai data ≤ t. Rank cross-sectional hanya memakai baris di tanggal yang sama.
Kolom keluaran berawalan f_ (masuk model); kolom bantu tanpa f_.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fitur yang juga diberi versi rank cross-sectional f_xs_<nama> (0 = terendah hari itu, 1 = tertinggi).
XS_RANK_FEATURES = [
    "f_ret5", "f_ret20", "f_mom_12_1", "f_mom_6_1", "f_dist_52w_high", "f_resid_ret5", "f_resid_ret21",
    "f_vol20", "f_idio_vol60", "f_max_ret21", "f_beta60", "f_volume_ratio", "f_rsi14", "f_dist_sma50",
    "f_log_value20", "f_atr_pct", "f_rs_sector20", "f_ear3", "f_days_since_earn",
]


def per_stock_alpha_features(g: pd.DataFrame) -> dict:
    """Dipanggil di dalam compute_stock_features (satu ticker, urut tanggal)."""
    c, h = g["close"], g["high"]
    r1 = c.pct_change()
    out = {
        "f_mom_12_1": c.shift(21) / c.shift(252) - 1,
        "f_mom_6_1": c.shift(21) / c.shift(126) - 1,
        "f_dist_52w_high": c / h.rolling(252, min_periods=126).max() - 1,
        "f_max_ret21": r1.rolling(21, min_periods=15).max(),
        "f_min_ret21": r1.rolling(21, min_periods=15).min(),
        "f_skew60": r1.rolling(60, min_periods=40).skew(),
    }
    return out


def add_market_relative(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Beta, volatilitas idiosinkratik, dan return residual. Butuh kolom date, ticker, close, idx_close, ret5, ret20,
    idx_ret5, idx_ret20 (dari regime). Semua rolling hanya ke belakang."""
    df = df.sort_values(["ticker", "date"]).copy()
    idx = df[["date", "idx_close"]].drop_duplicates("date").set_index("date")["idx_close"].sort_index()
    m1 = idx.pct_change()
    df["_r"] = df.groupby("ticker", sort=False)["close"].pct_change()
    df["_m"] = df["date"].map(m1)
    df["_rm"] = df["_r"] * df["_m"]
    df["_m2"] = df["_m"] ** 2
    g = df.groupby("ticker", sort=False)
    minp = max(20, window * 2 // 3)

    def roll(col):
        return g[col].transform(lambda s: s.rolling(window, min_periods=minp).mean())

    e_r, e_m, e_rm, e_m2 = roll("_r"), roll("_m"), roll("_rm"), roll("_m2")
    var_m = (e_m2 - e_m ** 2).where(lambda v: v > 1e-12)
    beta = ((e_rm - e_r * e_m) / var_m).clip(-1.0, 4.0)
    df["f_beta60"] = beta
    b = beta.fillna(1.0)
    df["_eps"] = df["_r"] - b * df["_m"]
    df["f_idio_vol60"] = df.groupby("ticker", sort=False)["_eps"].transform(
        lambda s: s.rolling(window, min_periods=minp).std())
    df["f_resid_ret5"] = df["ret5"] - b * df["idx_ret5"]
    df["f_resid_ret21"] = df["ret20"] - b * df["idx_ret20"]
    return df.drop(columns=["_r", "_m", "_rm", "_m2", "_eps"])


def add_cross_sectional_ranks(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = [c for c in (cols or XS_RANK_FEATURES) if c in df.columns]
    if not cols:
        return df
    elig = df["is_tradeable"].astype(bool) if "is_tradeable" in df else pd.Series(True, index=df.index)
    sub = df[cols].astype(float)
    sub.loc[~elig.to_numpy()] = np.nan                 # rank hanya di antara saham yang bisa ditransaksikan hari itu
    ranked = sub.groupby(df["date"]).rank(pct=True)
    for c in cols:
        df["f_xs_" + c[2:]] = ranked[c]
    return df
