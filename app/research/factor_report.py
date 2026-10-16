"""Laporan riset faktor: seberapa kuat SETIAP fitur memprediksi return residual 5 hari ke depan.

Cara kerja quant: sebelum memercayai model, ukur bukti tiap sinyal secara terpisah.
- Target = y_xs_5d (return saham − beta × S&P 500), jadi faktor tidak "menang" hanya karena beta atau arah pasar.
- Rank IC harian = korelasi Spearman nilai faktor vs target di antara saham pada tanggal yang sama.
- Hanya setiap h-hari (h = horizon) yang dipakai → label tidak tumpang tindih sehingga t-stat tidak menggelembung.
- Dilaporkan per periode (IC 12 bulan terakhir vs seluruh histori) untuk melihat apakah sinyal masih hidup.

|t| ≥ 2 ≈ signifikan secara statistik; IC 0,02–0,05 sudah tergolong sinyal yang berguna di saham likuid AS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import get


def _ic_series(sub: pd.DataFrame, feature: str, target: str, min_names: int) -> pd.Series:
    d = sub[["date", feature, target]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    cnt = d.groupby("date")[feature].transform("size")
    d = d[cnt >= min_names]
    if d.empty:
        return pd.Series(dtype=float)
    r = d.groupby("date")[[feature, target]].rank(pct=True)
    r["date"] = d["date"]
    return r.groupby("date").apply(lambda g: g[feature].corr(g[target]), include_groups=False).dropna()


def _stats(ic: pd.Series) -> dict:
    n = len(ic)
    if n < 5:
        return {"ic_mean": np.nan, "ic_tstat": np.nan, "ic_hit": np.nan, "n_dates": n}
    sd = ic.std(ddof=1)
    return {"ic_mean": float(ic.mean()), "ic_tstat": float(ic.mean() / sd * np.sqrt(n)) if sd > 0 else np.nan,
            "ic_hit": float((ic > 0).mean()), "n_dates": n}


def factor_ic_table(df: pd.DataFrame, cfg: dict, features: list[str] | None = None, min_names: int = 30) -> pd.DataFrame:
    h = int(cfg["labels"]["SWING_HORIZON"])
    target = f"y_xs_{h}d" if f"y_xs_{h}d" in df else f"y_ret_{h}d"
    sub = df[df["is_tradeable"].astype(bool)] if "is_tradeable" in df else df
    dates = np.sort(sub["date"].unique())
    keep = set(dates[::h])                                # non-overlapping
    sub = sub[sub["date"].isin(keep)]
    feats = features or sorted(c for c in df.columns if c.startswith("f_"))
    last_year = pd.Timestamp(dates[-1]) - pd.DateOffset(months=12) if len(dates) else None
    rows = []
    for f in feats:
        ic = _ic_series(sub, f, target, min_names)
        if ic.empty:
            continue
        full = _stats(ic)
        recent = _stats(ic[ic.index >= last_year]) if last_year is not None else {}
        rows.append({"feature": f, **full, "ic_mean_12m": recent.get("ic_mean"), "ic_tstat_12m": recent.get("ic_tstat"),
                     "coverage": float(sub[f].notna().mean())})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["abs_t"] = t["ic_tstat"].abs()
    t = t.sort_values("abs_t", ascending=False).drop(columns="abs_t").reset_index(drop=True)
    t.attrs["target"] = target
    return t


def run_factor_report(ctx, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hitung, simpan ke tabel reports (kind factor_ic), kembalikan tabelnya."""
    from app.pipeline.market_data import build_dataset
    cfg = ctx.cfg
    if df is None:
        df = build_dataset(cfg, ctx.repo, None, labels=True)
    t = factor_ic_table(df, cfg, min_names=int(get(cfg, "research.factor_min_names", 30) or 30))
    if len(t):
        d = str(pd.Timestamp(df["date"].max()).date())
        ctx.reports.save(d, "factor_ic", f"factor_ic_{d.replace('-', '')}.csv", t.round(4).to_csv(index=False).encode())
    return t
