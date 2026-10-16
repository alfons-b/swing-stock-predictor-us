"""Evaluasi prediksi historis (§40): setelah horizon lewat → actual return, MFE, MAE, hit SL/TP1/TP2, benar/salah."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.database.repository import Repository, now_utc
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def evaluate_one(p, fut: pd.DataFrame, cfg: dict) -> dict | None:
    """p: baris prediksi; fut: bar SETELAH tanggal prediksi (urut tanggal)."""
    horizons = cfg["labels"]["HORIZONS"]
    hold = int(cfg["strategy"]["max_holding_days"])
    need = max(max(horizons), hold)
    if len(fut) < need:
        return None
    c0 = float(p.close_price)
    ret = {h: float(fut["close"].iloc[h - 1] / c0 - 1) for h in horizons}
    H = cfg["labels"]["SWING_HORIZON"]
    win = fut.iloc[:H]
    mfe, mae = float(win["high"].max() / c0 - 1), float(win["low"].min() / c0 - 1)
    filled = bool(pd.notna(p.entry_high) and fut["low"].iloc[0] <= p.entry_high)
    hit_stop = hit1 = hit2 = False
    outcome = "NOT_FILLED" if not filled else "TIME_EXIT"
    if filled:
        for b in fut.iloc[:hold].itertuples(index=False):
            if pd.notna(p.stop_loss) and b.low <= p.stop_loss:  # candle sama → SL dulu (konservatif)
                hit_stop, outcome = True, "STOP" if not hit1 else "TP1_THEN_STOP"
                break
            if pd.notna(p.tp1) and b.high >= p.tp1:
                hit1, outcome = True, "TP1"
            if pd.notna(p.tp2) and b.high >= p.tp2:
                hit2, outcome = True, "TP2"
                break
    bull, bear = cfg["labels"]["bull_threshold"], cfg["labels"]["bear_threshold"]
    actual = 2 if ret[H] > bull else 0 if ret[H] < bear else 1
    probs = [p.prob_bearish, p.prob_neutral, p.prob_bullish]
    correct = int(np.nanargmax(probs)) == actual if not all(pd.isna(probs)) else None
    return {"prediction_id": int(p.id), "evaluated_at": now_utc(),
            **{f"actual_return_{h}d": ret.get(h) for h in (3, 5, 10)}, "mfe": mfe, "mae": mae,
            "entry_filled": filled, "hit_stop": hit_stop, "hit_tp1": hit1, "hit_tp2": hit2,
            "actual_class": actual, "prediction_correct": correct, "outcome": outcome}


def evaluate_predictions(cfg: dict, repo: Repository, prices: pd.DataFrame | None = None) -> dict:
    pend = repo.pending_evaluations()
    if pend.empty:
        return {"pending": 0, "evaluated": 0}
    if prices is None:
        prices = repo.load_prices(start=pend["prediction_date"].min())
    by_t = {t: g.sort_values("date").reset_index(drop=True) for t, g in prices.groupby("ticker")}
    rows = []
    for p in pend.itertuples(index=False):
        g = by_t.get(p.ticker)
        if g is None:
            continue
        r = evaluate_one(p, g[g["date"] > p.prediction_date], cfg)
        if r:
            rows.append(r)
    if rows:
        repo.save_evaluations(pd.DataFrame(rows))
    log.info("Evaluasi prediksi: %d dievaluasi, %d masih menunggu horizon", len(rows), len(pend) - len(rows),
             extra={"persist": True})
    return {"pending": int(len(pend) - len(rows)), "evaluated": len(rows)}


def live_calibration(repo: Repository, days: int = 120, bins: int = 10, min_n: int = 300) -> dict:
    """Monitoring kalibrasi prediksi LIVE (out-of-sample sejati): ECE & Brier prob_bullish vs kelas aktual,
    plus hit rate BUY per regime. n < min_n → INSUFFICIENT_DATA (tidak disimpulkan)."""
    since = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    df = repo.db.query_df("SELECT p.prob_bullish, p.decision, p.market_regime, e.actual_class, e.outcome "
                          "FROM prediction_evaluations e JOIN predictions p ON p.id = e.prediction_id "
                          "WHERE p.prediction_date >= ? AND e.actual_class IS NOT NULL", (since,))
    out = {"window_days": days, "n": int(len(df))}
    if len(df) < min_n:
        out["status"] = "INSUFFICIENT_DATA"
        return out
    p = pd.to_numeric(df["prob_bullish"], errors="coerce")
    y = (pd.to_numeric(df["actual_class"], errors="coerce") == 2).astype(float)
    ok = p.notna()
    p, y = p[ok], y[ok]
    b = np.minimum((p * bins).astype(int), bins - 1)
    ece = float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any()))
    out.update(status="OK", ece_bullish=round(ece, 4), brier_bullish=round(float(((p - y) ** 2).mean()), 4),
               mean_prob_bullish=round(float(p.mean()), 4), realized_bullish_rate=round(float(y.mean()), 4))
    buys = df[df["decision"] == "BUY"]
    if len(buys):
        win = buys["outcome"].isin(["TP1", "TP2", "TP1_THEN_STOP"])
        out["buy_hit_rate_by_regime"] = {str(k): {"n": int(len(g)), "tp_hit_rate": round(float(win[g.index].mean()), 3)}
                                         for k, g in buys.groupby(buys["market_regime"].fillna("UNKNOWN"))}
    return out
