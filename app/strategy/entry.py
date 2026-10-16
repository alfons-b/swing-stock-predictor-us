"""Entry zone (P): Entry Low / Ideal / High, dibulatkan ke tick harga AS ($0,01).

Breakout → zona dari level breakout (retest) sampai sedikit di atas close (batas kejar `max_chase_pct`).
Pullback → zona di sekitar close dengan bias ke bawah (beli pada kelemahan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy.setups import BREAKOUT_SETUPS, PULLBACK_SETUPS
from app.utils.market_rules import round_to_tick_vec


def compute_entry(df: pd.DataFrame, setups: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    st = cfg["strategy"]
    c, atr = df["close"], df["atr14"]
    w = st.get("entry_zone_atr", 0.25) * atr
    chase = c * (1 + st.get("max_chase_pct", 0.02))
    is_bo = setups["setup_type"].isin(BREAKOUT_SETUPS)
    is_pb = setups["setup_type"].isin(PULLBACK_SETUPS)
    ideal = c.copy()
    low = np.where(is_bo, np.maximum(df["res20"].fillna(c - w), c - w), c - w)
    low = np.where(is_pb, c - 1.5 * w, low)
    high = np.minimum(c + np.where(is_pb, 0.5 * w, w), chase)
    out = pd.DataFrame(index=df.index)
    out["entry_low"] = round_to_tick_vec(np.minimum(low, ideal), "down")
    out["entry_ideal"] = round_to_tick_vec(ideal.to_numpy(), "nearest")
    out["entry_high"] = round_to_tick_vec(np.maximum(high, ideal), "up")
    return out
