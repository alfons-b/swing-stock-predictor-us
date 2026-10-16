"""Margin of safety = (fair_value − harga) / fair_value, dan klasifikasi status valuasi.

Dua versi: `base` (nilai wajar tengah) dan `conservative` (nilai wajar BAWAH rentang).
Status:
  DEEP_VALUE        MoS base ≥ deep_value  DAN  MoS konservatif ≥ 0  (murah bahkan di skenario pesimis)
  UNDERVALUED       MoS base ≥ undervalued
  FAIRLY_VALUED     overvalued < MoS base < undervalued
  OVERVALUED        MoS base ≤ overvalued
  INSUFFICIENT_DATA nilai wajar tidak dapat dihitung secara layak
Keyakinan rendah (confidence LOW) TIDAK pernah menghasilkan DEEP_VALUE — diturunkan ke UNDERVALUED.
"""
from __future__ import annotations


def margin_of_safety(fair_value: float | None, price: float | None) -> float | None:
    if fair_value is None or price is None or fair_value <= 0 or price <= 0:
        return None
    return (fair_value - price) / fair_value


def classify(mos_base: float | None, mos_conservative: float | None, confidence: str | None, thresholds: dict) -> str:
    if mos_base is None:
        return "INSUFFICIENT_DATA"
    deep = float(thresholds.get("deep_value", 0.40))
    under = float(thresholds.get("undervalued", 0.20))
    over = float(thresholds.get("overvalued", -0.15))
    if mos_base >= deep and (mos_conservative or -1) >= 0 and confidence != "LOW":
        return "DEEP_VALUE"
    if mos_base >= under:
        return "UNDERVALUED"
    if mos_base <= over:
        return "OVERVALUED"
    return "FAIRLY_VALUED"
