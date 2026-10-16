"""Market regime (bagian F): S&P 500 trend, momentum, volatilitas, dan breadth pasar.

Skor komposit di [-1, 1] → 5 kelas. Rule-based & transparan (bukan ML) agar bisa dipakai
sebagai risk filter yang bisa dijelaskan. Semua komponen hanya memakai data <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features import indicators as ind

REGIMES = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
REGIME_CODE = {r: i - 2 for i, r in enumerate(REGIMES)}


def compute_breadth(stock_feats: pd.DataFrame) -> pd.DataFrame:
    ok = ~stock_feats["is_suspended"]
    d = stock_feats[ok]
    return pd.DataFrame({
        "breadth_above_sma50": (d["close"] > d["sma50"]).groupby(d["date"]).mean(),
        "breadth_above_sma200": (d["close"] > d["sma200"]).groupby(d["date"]).mean(),
        "breadth_adv_ratio": (d["ret1"] > 0).groupby(d["date"]).mean(),
        "breadth_new_high20": (d["breakout20"] > 0).groupby(d["date"]).mean(),
    })


def compute_market_regime(index: pd.DataFrame, breadth: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rc = cfg.get("regime", {})
    w = rc.get("weights", {"trend": 0.35, "momentum": 0.25, "breadth": 0.2, "volatility": 0.2})
    th = rc.get("thresholds", {"strong_bull": 0.45, "bull": 0.15, "bear": -0.15, "strong_bear": -0.45})
    ix = index.sort_values("date").set_index("date")
    c = ix["close"]
    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)
    r = pd.DataFrame(index=ix.index)
    r["idx_close"] = c
    for n in (5, 20, 60):
        r[f"idx_ret{n}"] = c / c.shift(n) - 1
    r["idx_dist_sma50"] = c / sma50 - 1
    r["idx_dist_sma200"] = c / sma200 - 1
    r["idx_vol20"] = c.pct_change().rolling(20, min_periods=20).std()
    r["idx_vol_pct"] = ind.rolling_pct_rank(r["idx_vol20"], 250)
    r = r.join(breadth, how="left")
    r["breadth_above_sma50"] = r["breadth_above_sma50"].ffill()

    trend = ((c > sma50).astype(float) + (c > sma200).astype(float) + (sma50 > sma200).astype(float)) / 1.5 - 1
    mom = np.clip(r["idx_ret20"] / 0.05, -1, 1)
    br = np.clip((r["breadth_above_sma50"].fillna(0.5) - 0.5) * 2.5, -1, 1)
    volp = -np.clip((r["idx_vol_pct"].fillna(0.5) - 0.5) * 2, -1, 1)
    score = w["trend"] * trend + w["momentum"] * mom + w["breadth"] * br + w["volatility"] * volp
    r["regime_score"] = score.where(sma200.notna())
    lab = np.select(
        [score >= th["strong_bull"], score >= th["bull"], score > th["bear"], score > th["strong_bear"]],
        ["STRONG_BULL", "BULL", "NEUTRAL", "BEAR"], default="STRONG_BEAR")
    r["market_regime"] = pd.Series(lab, index=r.index).where(r["regime_score"].notna(), "NEUTRAL")
    r["regime_code"] = r["market_regime"].map(REGIME_CODE)
    return r.reset_index()
