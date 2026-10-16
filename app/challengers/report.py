"""Laporan model challenger (tanpa torch): dipakai CLI `challenger-report` dan dashboard (halaman Model Lab)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.challengers import MODEL_LABELS, MODEL_NAMES

COLUMNS = ["model", "versi", "periode test", "n", "AUC bullish", "AUC naik", "log loss", "ECE", "ECE mentah", "top-k net",
           "aturan ranking", "top-k aturan", "CI95 top-k aturan", "p top-k aturan", "IC harian", "IC t", "cakupan 80%"]


def _j(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return v or {}


def comparison_table(repo) -> pd.DataFrame:
    """Satu baris per model challenger (versi terbaru) + model aktif + baseline, metrik pada periode test tersegel."""
    reg = repo.challenger_models()
    rows, base, active = [], {}, None
    for name in MODEL_NAMES:
        r = reg[reg["model_name"] == name] if len(reg) else reg
        if r is None or r.empty:
            continue
        r = r.iloc[0]
        m = _j(r["metrics"])
        t = m.get("test", {})
        base = base or m.get("baselines", {})
        sk = (m.get("significance") or {}).get("topk") or {}
        ci = f"{sk['ci_low']:+.2%} … {sk['ci_high']:+.2%}" if sk.get("ci_low") is not None else None
        active = active or m.get("active_model")
        rows.append({"model": MODEL_LABELS[name], "versi": r["version"],
                     "periode test": f"{t.get('eval_start')}..{t.get('eval_end')}", "n": t.get("n"),
                     "AUC bullish": t.get("roc_auc_bullish"), "log loss": t.get("log_loss"), "ECE": t.get("ece_bullish"),
                     "ECE mentah": t.get("ece_bullish_raw"), "top-k net": t.get("topk_mean_net_return"),
                     "aturan ranking": t.get("ranking_rule"), "top-k aturan": t.get("topk_by_rule_net"),
                     "CI95 top-k aturan": ci, "p top-k aturan": sk.get("p_value"),
                     "IC harian": t.get("ic_mean"), "IC t": t.get("ic_t"),
                     "AUC naik": t.get("auc_up"), "cakupan 80%": t.get("coverage_80")})
    if active:
        rows.append({"model": f"Model aktif ({active.get('version')})", "versi": "—",
                     "periode test": f"{active.get('eval_start')}..{active.get('eval_end')}",
                     "AUC bullish": active.get("roc_auc_bullish"), "log loss": active.get("log_loss"),
                     "top-k net": active.get("topk_mean_net_return")})
    for key, label in (("momentum", "Baseline momentum"), ("reversal", "Baseline reversal")):
        b = (base or {}).get(key)
        if b:
            rows.append({"model": label, "versi": "—", "top-k net": b.get("topk_mean_net_return"),
                         "IC harian": b.get("ic_mean"), "IC t": b.get("ic_t")})
    if base and base.get("universe_mean_net_return") is not None:
        rows.append({"model": "Rata-rata universe (beli semua)", "versi": "—", "top-k net": base["universe_mean_net_return"]})
    return pd.DataFrame(rows, columns=COLUMNS)


def realized_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """(ticker, date) → return sederhana `horizon` hari bursa ke depan dari close di database."""
    p = prices.sort_values(["ticker", "date"])
    p = p.assign(realized=p.groupby("ticker")["close"].shift(-horizon) / p["close"] - 1)
    return p[["ticker", "date", "realized"]].dropna()


def live_evaluation(forecasts: pd.DataFrame, prices: pd.DataFrame, horizon: int, bull_threshold: float) -> pd.DataFrame:
    """Evaluasi prakiraan PRODUKSI yang horizonnya sudah lewat (out-of-sample sungguhan, setelah model dilatih)."""
    if forecasts is None or forecasts.empty or prices is None or prices.empty:
        return pd.DataFrame()
    rr = realized_returns(prices, horizon)
    f = forecasts.merge(rr, left_on=["ticker", "forecast_date"], right_on=["ticker", "date"], how="inner")
    out = []
    from sklearn.metrics import roc_auc_score
    for m, g in f.groupby("model_name"):
        y_bull = (g["realized"] > bull_threshold).astype(int)
        ic = g.groupby("forecast_date").apply(lambda x: x["exp_return"].corr(x["realized"], method="spearman")
                                              if len(x) >= 20 else np.nan, include_groups=False).dropna()
        out.append({"model": MODEL_LABELS.get(m, m), "hari": int(g["forecast_date"].nunique()), "n": int(len(g)),
                    "AUC bullish": float(roc_auc_score(y_bull, g["prob_bullish"])) if 0 < y_bull.sum() < len(y_bull) else None,
                    "IC harian": float(ic.mean()) if len(ic) else None,
                    "cakupan 80%": float(((g["realized"] >= g["q10"]) & (g["realized"] <= g["q90"])).mean())})
    return pd.DataFrame(out)
