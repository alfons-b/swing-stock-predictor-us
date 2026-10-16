"""Position sizing (R) berbasis risiko, dibulatkan ke jumlah saham utuh (pasar AS: lot = 1 saham).

shares = (portfolio × risk_per_trade × regime_risk_mult) / (entry − stop)
dibatasi: max_position_pct portofolio dan MAX_POSITION_PCT_OF_ADV × rata-rata nilai transaksi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_position_size(df: pd.DataFrame, entry: pd.DataFrame, stops: pd.DataFrame, cfg: dict,
                          portfolio_value: float | None = None) -> pd.DataFrame:
    pf = cfg["portfolio"]
    lot = pf.get("board_lot", 1)
    pv = portfolio_value if portfolio_value is not None else pf["portfolio_value"]
    rules = cfg.get("regime", {}).get("rules", {})
    mult = pd.to_numeric(df["market_regime"].astype(object).map(lambda r: rules.get(r, {}).get("risk_mult", 1.0)),
                         errors="coerce").fillna(1.0)
    e = entry["entry_ideal"]
    risk_ps = (e - stops["stop_loss"]).clip(lower=1e-9)
    risk_amount = pv * pf["RISK_PER_TRADE"] * mult
    raw = risk_amount / risk_ps
    cap_pf = pf.get("max_position_pct", 0.2) * pv / e
    cap_adv = cfg["liquidity"].get("MAX_POSITION_PCT_OF_ADV", 0.05) * df["avg_value20"] / e
    shares = np.minimum.reduce([raw.to_numpy(), cap_pf.to_numpy(), cap_adv.fillna(0).to_numpy()])
    lots = np.floor(np.nan_to_num(shares) / lot).astype(int)
    out = pd.DataFrame(index=df.index)
    out["position_lots"] = lots
    out["position_size"] = lots * lot
    out["position_value"] = out["position_size"] * e
    out["risk_amount"] = out["position_size"] * risk_ps
    out["risk_budget"] = risk_amount
    out["capital_required"] = out["position_value"]
    out["estimated_loss"] = out["risk_amount"]  # kerugian bila stop loss tersentuh (sebelum biaya/slippage)
    out["size_capped_by"] = np.select([raw <= np.minimum(cap_pf, cap_adv.fillna(0)), cap_pf <= cap_adv.fillna(0)],
                                      ["risk", "max_position_pct"], "liquidity_adv")
    return out
