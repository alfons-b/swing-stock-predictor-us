"""Take profit (O) & risk/reward (Q).

TP1/TP2 = entry + k × risk (TP_MULTIPLIER). Resistance-aware: bila resistance berikutnya
berada di antara TP1 dan TP2, TP2 dipangkas ke bawah resistance; bila resistance di bawah
TP1, TP1 dipangkas — RR turun dan filter RR akan menolak trade yang memang tidak layak.
RR = rata-rata tertimbang (partial exit di TP1, sisa di TP2); bila resistance < TP1, RR = RR ke resistance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.market_rules import round_to_tick_vec, tick_size


def next_resistance(df: pd.DataFrame, entry: pd.Series) -> pd.Series:
    levels = df[["res20", "res50", "res120"]].to_numpy()
    e = entry.to_numpy()[:, None]
    above = np.where(levels > e * 1.005, levels, np.nan)
    with np.errstate(all="ignore"):
        r = np.nanmin(np.where(np.isnan(above), np.inf, above), axis=1)
    return pd.Series(np.where(np.isfinite(r), r, np.nan), index=df.index)


def compute_targets(df: pd.DataFrame, entry: pd.DataFrame, stops: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    st = cfg["strategy"]
    k1, k2 = st["TP_MULTIPLIER"]
    frac = st.get("tp_partial_fraction", 0.5)
    e = entry["entry_ideal"]
    risk = e - stops["stop_loss"]
    tp1 = e + k1 * risk
    tp2 = e + k2 * risk
    res = next_resistance(df, e)
    ticks = e.map(tick_size)
    res_below_tp2 = res.notna() & (res < tp2)
    res_below_tp1 = res.notna() & (res < tp1)
    tp2_adj = np.where(res_below_tp2 & ~res_below_tp1, res - ticks, tp2)
    tp1_adj = np.where(res_below_tp1, res - ticks, tp1)
    tp2_adj = np.where(res_below_tp1, np.maximum(tp2, tp1_adj), tp2_adj)  # resistance pertama < TP1 → TP2 tetap formula
    out = pd.DataFrame(index=df.index)
    out["take_profit_1"] = round_to_tick_vec(np.asarray(tp1_adj, float), "down")
    out["take_profit_2"] = round_to_tick_vec(np.maximum(np.asarray(tp2_adj, float), out["take_profit_1"] + ticks), "down")
    out["next_resistance"] = res
    out["tp_warning"] = np.select([res_below_tp1, res_below_tp2],
                                  ["Resistance terdekat di bawah TP1 — target dipangkas", "Resistance terdekat di bawah TP2 — TP2 dipangkas"], "")
    out["risk_per_share"] = risk
    out["potential_profit_tp1"] = out["take_profit_1"] - e
    out["potential_profit_tp2"] = out["take_profit_2"] - e
    rr1 = out["potential_profit_tp1"] / risk.replace(0, np.nan)
    rr2 = out["potential_profit_tp2"] / risk.replace(0, np.nan)
    out["rr_tp1"], out["rr_tp2"] = rr1, rr2
    # Konservatif: bila resistance terdekat < TP1, target realistis hanya sampai resistance itu.
    out["risk_reward"] = np.where(res_below_tp1, rr1, frac * rr1 + (1 - frac) * rr2)
    return out
