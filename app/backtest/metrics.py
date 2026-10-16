"""Metrik strategi (M). Semua dihitung SETELAH biaya transaksi."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # hari bursa NYSE per tahun


def max_consecutive(series: pd.Series) -> int:
    best = cur = 0
    for v in series:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial: float, rf: float = 0.0) -> dict:
    eq = equity.set_index("date")["equity"]
    if len(eq) < 2:
        return {"number_of_trades": int(len(trades))}
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / initial - 1
    cagr = (eq.iloc[-1] / initial) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    ex = rets - rf / TRADING_DAYS
    sharpe = np.sqrt(TRADING_DAYS) * ex.mean() / ex.std() if ex.std() > 0 else 0.0
    downside = ex[ex < 0].std()
    sortino = np.sqrt(TRADING_DAYS) * ex.mean() / downside if downside and downside > 0 else 0.0
    annual = eq.resample("YE").last().pct_change()
    annual.iloc[0] = eq.resample("YE").last().iloc[0] / initial - 1
    m = {
        "start": eq.index[0].strftime("%Y-%m-%d"), "end": eq.index[-1].strftime("%Y-%m-%d"),
        "initial_capital": initial, "final_equity": float(eq.iloc[-1]),
        "total_return": float(total), "cagr": float(cagr), "annual_return_mean": float(annual.mean()),
        "annual_returns": {str(k.year): float(v) for k, v in annual.items()},
        "max_drawdown": float(mdd), "sharpe": float(sharpe), "sortino": float(sortino),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else float("nan"),
        "exposure": float(equity["exposure"].mean()), "number_of_trades": int(len(trades)),
    }
    if len(trades):
        pnl = trades["net_pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        r = trades["net_return"]
        m.update({
            "win_rate": float((pnl > 0).mean()), "loss_rate": float((pnl <= 0).mean()),
            "average_win_pct": float(r[pnl > 0].mean()) if len(wins) else 0.0,
            "average_loss_pct": float(r[pnl <= 0].mean()) if len(losses) else 0.0,
            "expectancy_pct": float(r.mean()), "expectancy_rp": float(pnl.mean()),
            "expectancy_R": float(trades["r_multiple"].mean()),
            "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
            "average_holding_days": float(trades["holding_days"].mean()),
            "max_consecutive_losses": int(max_consecutive(pnl <= 0)),
            "turnover_annual": float(trades["entry_value"].sum() * 2 / eq.mean() / years),
            "recovery_factor": float((eq.iloc[-1] - initial) / abs((eq - eq.cummax()).min())) if mdd < 0 else float("nan"),
            "total_fees": float(trades["fees"].sum()),
            "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
        })
    return m
