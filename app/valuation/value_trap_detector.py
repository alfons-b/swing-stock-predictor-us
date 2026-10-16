"""Deteksi value trap: saham tampak murah karena bisnisnya memburuk, bukan karena salah harga.

Setiap pemeriksaan bernilai: True (tanda bahaya), False (aman), atau None (tidak dapat dinilai — data tidak ada).
risk:
  HIGH    poin ≥ 4, atau ada tanda berat (ekuitas negatif / rugi + utang naik / FCF negatif berulang + coverage < 1.5)
  MEDIUM  poin 2–3
  LOW     poin ≤ 1
  UNKNOWN pemeriksaan yang dapat dinilai < min_checks (data tidak cukup — BUKAN berarti aman)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.valuation.valuation_metrics import FINANCIAL_TYPES


def _series(hist, key):
    return [h.get(key) for h in hist]


def _declining(vals, n=2):
    v = [x for x in vals if x is not None]
    if len(v) < n + 1:
        return None
    tail = v[-(n + 1):]
    return all(b < a for a, b in zip(tail, tail[1:]))


def detect(snap: dict | None, metrics: dict, stype: str, prices: pd.DataFrame | None = None,
           as_of=None, stale_days: int = 450, min_checks: int = 4) -> dict:
    hist = (snap or {}).get("fy_history") or []
    it = (snap or {}).get("items") or {}
    checks: dict[str, tuple[bool | None, int, str]] = {}

    def chk(name, flag, points, text):
        checks[name] = (None if flag is None else bool(flag), points, text)

    chk("revenue_decline_2y", _declining(_series(hist, "revenue")), 1, "pendapatan turun 2 tahun berturut-turut")
    chk("earnings_decline_2y", _declining(_series(hist, "net_income")), 1, "laba bersih turun 2 tahun berturut-turut")
    ni = it.get("net_income")
    chk("loss_making", None if ni is None else ni <= 0, 2, "laba bersih ≤ 0 (rugi)")
    mc = metrics.get("margin_change")
    chk("margin_compression", None if mc is None else mc < -0.03, 1, "margin operasi menyusut > 3 poin persen")
    eq = it.get("total_equity")
    chk("negative_equity", None if eq is None else eq <= 0, 3, "ekuitas negatif")
    eqq = metrics.get("earnings_quality")
    chk("low_cash_conversion", None if eqq is None else eqq < 0.5, 1, "arus kas operasi < 50% laba (kualitas laba rendah)")
    if stype not in FINANCIAL_TYPES:
        lc = metrics.get("leverage_change")
        de = metrics.get("debt_to_equity")
        chk("leverage_rising", None if lc is None else lc > 0.3, 1, "rasio utang/ekuitas naik > 0,3 dalam setahun")
        chk("high_leverage", None if de is None else de > 2.0, 1, "utang/ekuitas > 2")
        ic = metrics.get("interest_coverage")
        chk("weak_interest_coverage", None if ic is None else ic < 2.0, 1, "EBIT < 2× beban bunga")
        fcfs = [(h.get("operating_cash_flow") - h.get("capex")) if None not in (h.get("operating_cash_flow"), h.get("capex"))
                else h.get("free_cash_flow") for h in hist[-3:]]
        fcfs = [f for f in fcfs if f is not None]
        chk("negative_fcf_repeated", None if len(fcfs) < 3 else sum(f < 0 for f in fcfs) >= 2, 1,
            "FCF negatif di ≥ 2 dari 3 tahun terakhir")
    divs = [d for d in _series(hist, "dividends_paid")[-2:]]
    chk("dividend_cut", None if len(divs) < 2 or None in divs or divs[0] <= 0 else divs[1] < 0.7 * divs[0], 1,
        "dividen dipotong > 30%")
    if prices is not None and len(prices) >= 200:
        c = prices["close"].astype(float).reset_index(drop=True)
        dd = c.iloc[-1] / c.iloc[-250:].max() - 1
        below = c.iloc[-1] < c.iloc[-200:].mean()
        chk("falling_knife", bool(below and dd < -0.40), 1, "harga di bawah MA200 dan turun > 40% dari puncak 52 minggu")
    else:
        chk("falling_knife", None, 1, "histori harga < 200 hari")
    known = (snap or {}).get("known_date")
    pe = (snap or {}).get("flow_period_end") or (snap or {}).get("period_end")
    if as_of is not None and pe is not None and not pd.isna(pe):
        chk("stale_fundamentals", (pd.Timestamp(as_of) - pd.Timestamp(pe)).days > stale_days, 1,
            f"laporan terbaru berakhir > {stale_days} hari lalu (data basi)")
    if "CHECK" in ((snap or {}).get("quality") or []):
        chk("data_quality", True, 1, "kualitas data laporan perlu dicek (satuan/mata uang/konsistensi)")

    evaluated = {k: v for k, v in checks.items() if v[0] is not None}
    flagged = {k: v for k, v in evaluated.items() if v[0]}
    points = sum(v[1] for v in flagged.values())
    severe = ("negative_equity" in flagged
              or ("loss_making" in flagged and "leverage_rising" in flagged)
              or ("negative_fcf_repeated" in flagged and "weak_interest_coverage" in flagged
                  and (metrics.get("interest_coverage") or 9) < 1.5))
    if len(evaluated) < min_checks:
        risk = "UNKNOWN"
    elif severe or points >= 4:
        risk = "HIGH"
    elif points >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {"risk": risk, "points": int(points), "n_checks": len(evaluated), "known_date": known,
            "reasons": [v[2] for v in flagged.values()],
            "not_evaluated": [k for k, v in checks.items() if v[0] is None],
            "checks": {k: v[0] for k, v in checks.items()}, "score": float(np.clip(points / 6, 0, 1) * 100)}
