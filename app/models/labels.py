"""Target label (K). SATU-SATUNYA modul yang boleh memakai data masa depan (shift negatif).
Kolom label diawali `y_` dan TIDAK PERNAH masuk ke daftar fitur (`f_*`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd



def _future_stack(s: pd.Series, h: int) -> np.ndarray:
    return np.column_stack([s.shift(-k).to_numpy() for k in range(1, h + 1)])


def add_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lc = cfg["labels"]
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        c = g["close"]
        for h in lc["HORIZONS"]:
            g[f"y_ret_{h}d"] = c.shift(-h) / c - 1
            hi = _future_stack(g["high"], h)
            lo = _future_stack(g["low"], h)
            complete = ~np.isnan(hi).any(axis=1)
            g[f"y_mfe_{h}d"] = np.where(complete, np.nanmax(np.where(np.isnan(hi), -np.inf, hi), axis=1) / c - 1, np.nan)
            g[f"y_mae_{h}d"] = np.where(complete, np.nanmin(np.where(np.isnan(lo), np.inf, lo), axis=1) / c - 1, np.nan)
        # hit TP sebelum SL (level generik berbasis ATR) untuk horizon utama
        h = lc["SWING_HORIZON"]
        tp = (c + lc["tp_atr_mult"] * g["atr14"]).to_numpy()
        sl = (c - lc["sl_atr_mult"] * g["atr14"]).to_numpy()
        hi = _future_stack(g["high"], h)
        lo = _future_stack(g["low"], h)
        res = np.full(len(g), np.nan)
        decided = np.zeros(len(g), bool)
        for k in range(h):
            hit_sl = (lo[:, k] <= sl) & ~decided
            hit_tp = (hi[:, k] >= tp) & ~decided & ~hit_sl  # candle sama → anggap SL dulu (pesimis)
            res[hit_sl] = 0
            res[hit_tp] = 1
            decided |= hit_sl | hit_tp
        complete = ~np.isnan(hi).any(axis=1) & np.isfinite(tp)
        res[complete & ~decided] = 0  # tidak mencapai TP dalam horizon
        res[~complete] = np.nan
        g["y_hit_tp"] = res
        g["y_hit_sl_first"] = np.where(complete, ((~np.isnan(res)) & (res == 0) & decided).astype(float), np.nan)
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df = add_class_label(df, cfg)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def class_thresholds(df: pd.DataFrame, cfg: dict, h: int | None = None):
    lc = cfg["labels"]
    h = h or lc["SWING_HORIZON"]
    if lc.get("mode", "fixed") == "vol_scaled":
        thr = lc.get("vol_k", 0.75) * df["vol20"] * np.sqrt(h)
        return thr, -thr
    return pd.Series(lc["bull_threshold"], index=df.index), pd.Series(lc["bear_threshold"], index=df.index)


def add_residual_returns(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """y_xs_{h}d = return saham h hari ke depan − beta(t) × return S&P 500 h hari ke depan (beta 60H, diketahui di t).
    y_xsz_{h}d = y_xs dibagi volatilitas harian 20H × √h (skala risiko — saham volatil tidak otomatis menang)."""
    if "idx_close" not in df:
        return df
    ix = df[["date", "idx_close"]].drop_duplicates("date").set_index("date")["idx_close"].sort_index()
    m_fwd = ix.shift(-h) / ix - 1
    beta = df["f_beta60"].fillna(1.0) if "f_beta60" in df else pd.Series(1.0, index=df.index)
    df[f"y_xs_{h}d"] = df[f"y_ret_{h}d"] - beta * df["date"].map(m_fwd)
    vol = df["vol20"].where(df["vol20"] > 0) if "vol20" in df else pd.Series(np.nan, index=df.index)
    vol = vol.fillna(vol.median() if vol.notna().any() else 0.02)
    df[f"y_xsz_{h}d"] = df[f"y_xs_{h}d"] / (vol * np.sqrt(h))
    return df


def add_class_label(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Label 3 kelas (0 bearish · 1 netral · 2 bullish).

    mode `xs_resid` (default sejak f2): peringkat cross-sectional per tanggal dari return RESIDUAL berskala risiko
      (y_xsz). Bullish = sepertiga teratas saham yang bisa ditransaksikan pada hari itu. Model belajar SAHAM MANA yang
      akan mengungguli saham lain setelah memperhitungkan pasar & volatilitas — bukan "pasar sedang naik" atau
      "saham ini volatil" (keduanya membuat label lama ±3% terlihat mudah ditebak: AUC 0,62 di produksi, tetapi
      top-5 hanya +0,23%/trade sebelum biaya).
    mode `fixed` / `vol_scaled`: label lama (return mentah > ambang)."""
    lc = cfg["labels"]
    h = lc["SWING_HORIZON"]
    r = df[f"y_ret_{h}d"]
    if lc.get("mode", "fixed") == "xs_resid" and "idx_close" in df:
        df = add_residual_returns(df, h)
        z = df[f"y_xsz_{h}d"]
        elig = df["is_tradeable"].astype(bool) if "is_tradeable" in df else pd.Series(True, index=df.index)
        q = float(lc.get("xs_quantile", 1 / 3))
        rank = z.where(elig).groupby(df["date"]).rank(pct=True)
        y = np.where(rank > 1 - q, 2, np.where(rank <= q, 0, 1)).astype(float)
        y[(z.isna() | rank.isna()).to_numpy()] = np.nan
        df["y_class"] = y
        return df
    up, dn = class_thresholds(df, cfg, h)
    y = np.where(r > up, 2, np.where(r < dn, 0, 1)).astype(float)
    y[r.isna().to_numpy()] = np.nan
    df["y_class"] = y
    return df
