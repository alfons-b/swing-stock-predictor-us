"""Aturan perdagangan pasar saham AS (NYSE / Nasdaq) yang relevan untuk eksekusi order.

- Tick size (Reg NMS Rule 612): harga ≥ $1,00 dikutip dalam kelipatan $0,01; di bawah $1,00 boleh $0,0001.
- Tidak ada "lot": order bisa 1 saham (round lot 100 hanya relevan untuk kuotasi, bukan untuk eksekusi ritel).
- Tidak ada batas auto-reject harian seperti ARA/ARB BEI. Pengaman volatilitas AS adalah LULD (jeda 5 menit)
  dan market-wide circuit breaker — keduanya tidak memblokir harga penutupan, jadi tidak dimodelkan di backtest.

Harga entry/SL/TP selalu dibulatkan ke tick valid agar order limit dapat dikirim apa adanya.
Nama fungsi sama dengan versi BEI (`round_to_tick`, `tick_size`, `shares_to_lots`) supaya modul strategi tidak berubah.
"""
from __future__ import annotations

import math

import numpy as np

BOARD_LOT = 1          # 1 saham — pasar AS tidak mewajibkan lot
_SUB_DOLLAR_TICK = 0.0001
_TICK = 0.01


def tick_size(price: float) -> float:
    return _SUB_DOLLAR_TICK if price is not None and price < 1.0 else _TICK


def _decimals(t: float) -> int:
    return 4 if t < 0.01 else 2


def round_to_tick(price: float, mode: str = "nearest") -> float:
    """mode: nearest | down | up. Hasil dibulatkan ke desimal tick agar bebas galat floating point."""
    if price is None or not np.isfinite(price) or price <= 0:
        return float("nan")
    t = tick_size(price)
    q = price / t
    if mode == "down":
        v = math.floor(q + 1e-9) * t
    elif mode == "up":
        v = math.ceil(q - 1e-9) * t
    else:
        v = round(q) * t
    # pembulatan di sekitar $1,00 bisa memindahkan harga ke band tick lain — ulangi sekali
    t2 = tick_size(v)
    if t2 != t:
        q2 = price / t2
        v = (math.floor(q2 + 1e-9) if mode == "down" else math.ceil(q2 - 1e-9) if mode == "up" else round(q2)) * t2
        t = t2
    return float(round(v, _decimals(t)))


round_to_tick_vec = np.vectorize(round_to_tick, otypes=[float])


def shares_to_lots(shares: float, lot: int = BOARD_LOT) -> int:
    """Jumlah unit order utuh. Dengan lot = 1 sama dengan jumlah saham utuh (dibulatkan ke bawah)."""
    if shares is None or not np.isfinite(shares) or shares <= 0:
        return 0
    return int(shares // max(1, int(lot)))
