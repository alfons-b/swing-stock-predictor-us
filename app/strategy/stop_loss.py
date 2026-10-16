"""Stop loss (N). Empat metode dihitung; metode dipilih per setup (config).

- atr       : entry − ATR × multiplier
- swing_low : low terendah N hari − buffer
- support   : invalidation level setup (support/level breakout) − buffer
- percent   : entry × (1 − pct)
Stop dipaksa minimal `min_stop_distance_pct` (hindari noise); bila lebih jauh dari
`max_stop_distance_pct` → setup ditolak (risiko per lembar terlalu besar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.market_rules import round_to_tick_vec


def compute_stops(df: pd.DataFrame, setups: pd.DataFrame, entry: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    st = cfg["strategy"]
    e = entry["entry_ideal"]
    atr = df["atr14"]
    buf = st.get("stop_buffer_atr", 0.2) * atr
    cand = pd.DataFrame(index=df.index)
    cand["atr"] = e - st["STOP_LOSS_ATR_MULTIPLIER"] * atr
    cand["swing_low"] = df["swing_low10"] - buf
    cand["support"] = setups["invalidation_level"] - buf
    cand["percent"] = e * (1 - st.get("stop_percent", 0.06))
    mapping = st.get("stop_method_by_setup", {})
    method = setups["setup_type"].map(lambda s: mapping.get(s, mapping.get("DEFAULT", "atr")))
    chosen = np.array([cand.at[i, m] for i, m in zip(df.index, method)], dtype=float)
    # fallback ke ATR bila level struktur tidak tersedia atau di atas entry
    bad = ~np.isfinite(chosen) | (chosen >= e.to_numpy())
    chosen = np.where(bad, cand["atr"].to_numpy(), chosen)
    method = method.where(~bad, "atr")
    min_d = e * (1 - st.get("min_stop_distance_pct", 0.02))
    widened = chosen > min_d.to_numpy()
    chosen = np.minimum(chosen, min_d.to_numpy())
    out = pd.DataFrame(index=df.index)
    out["stop_loss"] = round_to_tick_vec(chosen, "down")
    out["stop_method"] = method.where(~widened, method + "+min_dist")
    out["stop_distance_pct"] = 1 - out["stop_loss"] / e
    out["stop_too_wide"] = out["stop_distance_pct"] > st.get("max_stop_distance_pct", 0.10)
    for k in cand:
        out[f"stop_{k}"] = cand[k]
    return out
