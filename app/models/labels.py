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


def add_class_label(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = cfg["labels"]["SWING_HORIZON"]
    r = df[f"y_ret_{h}d"]
    up, dn = class_thresholds(df, cfg, h)
    y = np.where(r > up, 2, np.where(r < dn, 0, 1)).astype(float)
    y[r.isna().to_numpy()] = np.nan
    df["y_class"] = y
    return df
