"""NO-TRADE conditions (U) dan keputusan akhir: BUY / WATCHLIST / WAIT / AVOID.

Filosofi analis skeptis: cari dulu alasan trade bisa GAGAL. BUY hanya bila tidak ada
satu pun alasan penolakan. Default sistem adalah WAIT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.fundamental_sentiment import fundamental_risk_flags

HARD_RISK = ["illiquid", "suspended", "short_history", "price_too_low", "corporate_action_risk",
             "fundamental_risk", "extreme_volatility"]


def probability_threshold(pred: pd.DataFrame, cfg: dict) -> pd.Series:
    """Ambang P(bullish). Default: base rate training + MIN_PROB_EDGE (lift di atas acak).
    Bila MIN_CONFIDENCE diisi angka, dipakai sebagai ambang absolut."""
    st = cfg["strategy"]
    if st.get("MIN_CONFIDENCE") is not None:
        return pd.Series(float(st["MIN_CONFIDENCE"]), index=pred.index)
    base = pred["base_rate_bullish"] if "base_rate_bullish" in pred else pd.Series(1 / 3, index=pred.index)
    return base.fillna(1 / 3) + st.get("MIN_PROB_EDGE", 0.05)


def confidence_level(pred: pd.DataFrame, cfg: dict, prob_min: pd.Series, horizon: int) -> pd.Series:
    st = cfg["strategy"]
    p, disp = pred["probability_bullish"], pred["prob_dispersion"]
    er = pred.get(f"expected_return_{horizon}d", pd.Series(0.0, index=pred.index))
    high = (p >= prob_min + 0.05) & (disp <= st["max_uncertainty"] / 2) & (er > 0) & (p > 2 * pred["probability_bearish"])
    med = (p >= prob_min) & (disp <= st["max_uncertainty"]) & (er > 0)
    return pd.Series(np.select([high, med], ["HIGH", "MEDIUM"], "LOW"), index=pred.index)


def apply_filters(df: pd.DataFrame, setups: pd.DataFrame, pred: pd.DataFrame, stops: pd.DataFrame,
                  targets: pd.DataFrame, sizing: pd.DataFrame, scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    st, sc = cfg["strategy"], cfg["scoring"]
    h = cfg["labels"]["SWING_HORIZON"]
    rules = cfg.get("regime", {}).get("rules", {})
    reg = df["market_regime"]
    reg_o = reg.astype(object)
    prob_add = pd.to_numeric(reg_o.map(lambda r: rules.get(r, {}).get("prob_add", 0.0)), errors="coerce").fillna(0.0)
    allow_lq = reg_o.map(lambda r: bool(rules.get(r, {}).get("allow_low_quality_breakout", True))).astype(bool)
    prob_min = probability_threshold(pred, cfg) + prob_add
    er = pred.get(f"expected_return_{h}d", pd.Series(0.0, index=df.index))

    checks = {
        "illiquid": ~df["is_liquid"],
        "suspended": df["is_suspended"],
        "short_history": df["days_listed"] < cfg["quality"]["min_history_days"],
        "price_too_low": df["close"] < cfg["quality"]["min_price"],
        "corporate_action_risk": df["ca_recent"],
        "fundamental_risk": fundamental_risk_flags(df, cfg),
        "extreme_volatility": df["atr_pct"] > st["max_atr_pct"],
        "setup_not_confirmed": ~setups["setup_valid"],
        "low_quality_breakout_in_weak_regime": setups["is_low_quality_breakout"] & ~allow_lq,
        "stop_too_wide": stops["stop_too_wide"],
        "risk_reward_below_min": targets["risk_reward"] < st["MIN_RISK_REWARD"],
        "probability_below_min": pred["probability_bullish"] < prob_min,
        "bearish_outweighs_bullish": pred["probability_bearish"] >= pred["probability_bullish"],
        "expected_return_not_positive": er <= 0,
        "model_uncertainty_high": pred["prob_dispersion"] > st["max_uncertainty"],
        "position_too_small": sizing["position_lots"] < 1,
        "score_below_buy_min": scores["final_score"] < sc["buy_score_min"],
    }
    C = pd.DataFrame({k: v.fillna(True).astype(bool) for k, v in checks.items()}, index=df.index)
    out = pd.DataFrame(index=df.index)
    out["reject_reasons"] = C.apply(lambda r: list(r.index[r.to_numpy()]), axis=1) if len(C) <= 5000 else \
        [list(C.columns[row]) for row in C.to_numpy()]
    out["n_reject"] = C.sum(axis=1)
    hard = C[HARD_RISK].any(axis=1)
    out["confidence"] = confidence_level(pred, cfg, prob_min, h)
    out["prob_min_required"] = prob_min
    buy = (out["n_reject"] == 0)
    bearish = C["bearish_outweighs_bullish"] & C["expected_return_not_positive"]
    watch = ~hard & ~buy & (setups["setup_valid"] | setups["setup_forming"]) & \
        (scores["final_score"] >= sc["watchlist_score_min"]) & ~bearish
    out["decision"] = np.select([buy, hard | bearish, watch], ["BUY", "AVOID", "WATCHLIST"], "WAIT")
    return out
