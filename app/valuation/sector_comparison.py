"""Perbandingan relatif terhadap emiten sejenis (peer) — kuartil multiple per kelompok.

Kelompok peer: subsektor → sektor → tipe sektor (BANK/COMMODITY/...). Kelompok dipakai hanya bila jumlah emiten
dengan multiple VALID ≥ min_peers. Tidak ada fallback ke "seluruh pasar" (membandingkan bank dengan tambang
menyesatkan) — bila tidak ada kelompok yang cukup, metode relatif = tidak valid.
Outlier dikendalikan dengan kuartil (bukan rata-rata) dan multiple ekstrem dibuang sebelum dihitung.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# batas wajar sebelum dihitung kuartil (di luar ini hampir selalu artefak data / laba sangat kecil)
SANE = {"per": (0.5, 100.0), "pbv": (0.05, 30.0), "ev_ebitda": (0.5, 60.0), "ev_sales": (0.02, 30.0), "p_fcf": (0.5, 100.0)}
GROUP_LEVELS = ("subsector", "sector", "sector_type")


def peer_quartiles(universe: pd.DataFrame, metric: str, min_peers: int = 5) -> dict:
    """{(level, group): {q25, q50, q75, n}} untuk satu multiple. `universe` berisi kolom ticker + GROUP_LEVELS + metric."""
    if metric not in universe:
        return {}
    lo, hi = SANE.get(metric, (-np.inf, np.inf))
    v = universe[[*GROUP_LEVELS, "ticker", metric]].copy()
    v = v[v[metric].between(lo, hi)]
    out = {}
    for level in GROUP_LEVELS:
        g = v.dropna(subset=[level]).groupby(level)[metric]
        for name, s in g:
            if len(s) >= min_peers:
                out[(level, name)] = {"q25": float(s.quantile(0.25)), "q50": float(s.quantile(0.5)),
                                      "q75": float(s.quantile(0.75)), "n": int(len(s))}
    return out


def peer_group(row: dict, quart: dict) -> tuple[str, str, dict] | None:
    """Kelompok paling spesifik yang memenuhi min_peers untuk emiten ini."""
    for level in GROUP_LEVELS:
        key = (level, row.get(level))
        if row.get(level) is not None and key in quart:
            return level, row.get(level), quart[key]
    return None


def peer_percentile(universe: pd.DataFrame, ticker: str, metric: str, level: str) -> float | None:
    """Persentil (0–1) multiple emiten di kelompoknya. Rendah = lebih murah dari peer."""
    if metric not in universe:
        return None
    row = universe[universe["ticker"] == ticker]
    if row.empty or pd.isna(row[metric].iloc[0]) or pd.isna(row[level].iloc[0]):
        return None
    lo, hi = SANE.get(metric, (-np.inf, np.inf))
    grp = universe[(universe[level] == row[level].iloc[0]) & universe[metric].between(lo, hi)][metric]
    if len(grp) < 2:
        return None
    return float((grp < row[metric].iloc[0]).mean())
