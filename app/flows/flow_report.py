"""Teks bagian SHORT FLOW (FINRA; slot foreign flow versi BEI) untuk STOCK RESEARCH REPORT dan ringkasan pasar."""
from __future__ import annotations

import math

import pandas as pd


def _f(v, pct=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "UNAVAILABLE"
    return f"{v:+.1%}" if pct else f"{v:+,.0f}"


def format_flow(row: dict | None, source_status: str = "FOREIGN_FLOW_UNAVAILABLE") -> list[str]:
    if not row or row.get("foreign_flow_status") in (None, "INSUFFICIENT_DATA", "FOREIGN_FLOW_UNAVAILABLE"):
        cov = row.get("flow_coverage_20") if row else None
        return [f"Status               : {row.get('foreign_flow_status') if row else source_status}",
                f"Cakupan data 20H     : {'-' if cov is None or pd.isna(cov) else f'{cov:.0%}'}",
                "  Short flow tidak dipakai dalam skor (bukan dianggap netral/nol)."]
    vt = row.get("flow_value_type_20", "ESTIMATED_VALUE")
    return [f"Status               : {row.get('foreign_flow_status')}  (skor {row.get('foreign_flow_score', 0):.0f}, "
            f"confidence {row.get('foreign_flow_confidence')})",
            f"Net non-short 5/20/60H: {_f(row.get('flow_net_shares_5'))} / {_f(row.get('flow_net_shares_20'))} / "
            f"{_f(row.get('flow_net_shares_60'))} saham (volume FINRA − 2×short)",
            f"Net ratio 20H vs base: {_f(row.get('flow_net_ratio_20'), True)}   (+ = short di bawah kebiasaan 120H)",
            f"Net nilai 20H        : $ {_f(row.get('flow_net_value_20'))} [{vt}]",
            f"Hari short < base 20H: {(row.get('flow_buy_days_ratio_20') or float('nan')):.0%}   streak {row.get('flow_streak', 0):+.0f}",
            f"Cakupan data 20H     : {row.get('flow_coverage_20', 0):.0%}"]
