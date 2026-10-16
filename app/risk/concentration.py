"""Batas konsentrasi daftar BUY (dipakai identik oleh scan live dan backtest).

1. Sektor: maksimal `research_scoring.max_per_sector_buy` BUY per sektor per tanggal (urut rank_score).
2. Korelasi: kandidat BUY yang korelasi return harian 60 hari-nya > `max_pair_correlation` terhadap kandidat
   BUY yang peringkatnya lebih tinggi diturunkan ke WATCHLIST (risiko ganda pada faktor yang sama).
Korelasi memakai return s.d. tanggal sinyal saja (tanpa data masa depan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import get


def apply_concentration_limits(sig: pd.DataFrame, cfg: dict, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    max_sector = get(cfg, "research_scoring.max_per_sector_buy")
    max_corr = get(cfg, "research_scoring.max_pair_correlation")
    window = int(get(cfg, "research_scoring.correlation_window", 60))
    if sig is None or sig.empty or (max_sector is None and max_corr is None) or "decision" not in sig:
        return sig
    buys = sig[sig["decision"] == "BUY"]
    if len(buys) < 2:
        return sig
    sig = sig.copy()
    rets = None
    if max_corr is not None and prices is not None and len(prices):
        p = prices[["date", "ticker", "close"]].drop_duplicates(["date", "ticker"])
        p = p[p["ticker"].isin(buys["ticker"].unique())]
        rets = p.pivot(index="date", columns="ticker", values="close").sort_index().pct_change(fill_method=None)
    for d, grp in buys.groupby("date"):
        order = grp.sort_values("rank_score", ascending=False)
        kept, per_sector = [], {}
        hist = None
        if rets is not None:
            hist = rets.loc[:d].tail(window)
        for idx, row in order.iterrows():
            reason = None
            sec = row.get("sector") if "sector" in row else None
            if sec is None or pd.isna(sec) or str(sec).strip().lower() in ("", "unknown", "none"):
                sec = None                                    # sektor tidak diketahui → tidak dibatasi sektor
            if max_sector is not None and sec is not None and per_sector.get(sec, 0) >= int(max_sector):
                reason = "sector_concentration"
            elif hist is not None and kept and row["ticker"] in hist:
                for k in kept:
                    if k not in hist:
                        continue
                    pair = hist[[row["ticker"], k]].dropna()
                    if len(pair) >= window * 0.8:
                        c = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])
                        if np.isfinite(c) and c > float(max_corr):
                            reason = "high_correlation_with_higher_ranked_buy"
                            break
            if reason:
                sig.at[idx, "decision"] = "WATCHLIST"
                sig.at[idx, "reject_reasons"] = list(sig.at[idx, "reject_reasons"] or []) + [reason]
            else:
                kept.append(row["ticker"])
                if sec is not None:
                    per_sector[sec] = per_sector.get(sec, 0) + 1
    return sig
