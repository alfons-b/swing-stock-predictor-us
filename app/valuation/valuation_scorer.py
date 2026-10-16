"""Skor 0–100: value_score (seberapa murah, disesuaikan keyakinan) dan business_quality_score.

Skor None = data tidak cukup (BUKAN nol). Komponen dipetakan linear antara batas buruk→baik lalu dirata-rata.
"""
from __future__ import annotations

import numpy as np

CONF_SHRINK = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.5}


def _lin(x, bad, good):
    if x is None or not np.isfinite(x):
        return None
    return float(np.clip((x - bad) / (good - bad), 0, 1) * 100)


def value_score(mos_base: float | None, mos_conservative: float | None, confidence: str | None,
                peer_percentile: float | None = None) -> float | None:
    """50 = wajar. MoS +50% → 100, −50% → 0; disusutkan ke 50 sesuai keyakinan; bonus kecil dari posisi vs peer."""
    if mos_base is None:
        return None
    raw = 50 + 100 * float(np.clip(0.7 * mos_base + 0.3 * (mos_conservative if mos_conservative is not None else mos_base),
                                   -0.5, 0.5))
    if peer_percentile is not None:
        raw = 0.85 * raw + 0.15 * (100 * (1 - peer_percentile))
    s = CONF_SHRINK.get(confidence or "LOW", 0.5)
    return float(np.clip(50 + (raw - 50) * s, 0, 100))


def quality_score(metrics: dict, stype: str, min_components: int = 3) -> tuple[float | None, dict]:
    fin = stype in ("BANK", "FINANCIAL")
    comp = {
        "roe": _lin(metrics.get("roe"), 0.0, 0.20),
        "net_margin": None if fin else _lin(metrics.get("net_margin"), 0.0, 0.15),
        "earnings_quality": None if fin else _lin(metrics.get("earnings_quality"), 0.3, 1.1),
        "leverage": None if fin else _lin(-(metrics.get("debt_to_equity") if metrics.get("debt_to_equity") is not None else np.nan), -2.0, -0.3),
        "interest_coverage": None if fin else _lin(metrics.get("interest_coverage"), 1.5, 8.0),
        "revenue_growth": _lin(metrics.get("revenue_cagr_3y", metrics.get("revenue_growth")), -0.05, 0.12),
        "roa": _lin(metrics.get("roa"), 0.0, 0.025) if fin else None,
        "margin_trend": _lin(metrics.get("margin_change"), -0.05, 0.03),
    }
    comp = {k: v for k, v in comp.items() if v is not None}
    if len(comp) < min_components:
        return None, comp
    return float(np.mean(list(comp.values()))), comp
