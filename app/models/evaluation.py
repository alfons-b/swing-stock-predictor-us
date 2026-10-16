"""Metrik prediktif & kalibrasi (AB). Semua menerima array, aman terhadap kelas yang hilang."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = P.argmax(axis=1)
    bull = (y == 2).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_bullish": float(precision_score(bull, (pred == 2).astype(int), zero_division=0)),
        "recall_bullish": float(recall_score(bull, (pred == 2).astype(int), zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(P, 1e-6, 1), labels=[0, 1, 2])),
        "brier_bullish": float(brier_score_loss(bull, P[:, 2])),
        "base_rate_bullish": float(bull.mean()),
    }
    if 0 < bull.sum() < len(bull):
        out["roc_auc_bullish"] = float(roc_auc_score(bull, P[:, 2]))
        out["pr_auc_bullish"] = float(average_precision_score(bull, P[:, 2]))
    else:
        out["roc_auc_bullish"] = out["pr_auc_bullish"] = float("nan")
    return out


def calibration_table(y_bull: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_bull, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


def expected_calibration_error(y_bull, p, bins: int = 10) -> float:
    t = calibration_table(y_bull, p, bins)
    return float((t["n"] * (t["predicted"] - t["actual"]).abs()).sum() / max(1, t["n"].sum()))


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 10:
        return {}
    ic = pd.Series(y).corr(pd.Series(yhat), method="spearman")
    return {"mae": float(np.mean(np.abs(y - yhat))), "directional_accuracy": float(np.mean(np.sign(y) == np.sign(yhat))),
            "spearman_ic": float(ic), "avg_predicted": float(np.mean(yhat)), "avg_actual": float(np.mean(y))}


def topk_trading_proxy(df: pd.DataFrame, score_col: str, ret_col: str, k: int, cost: float) -> dict:
    """Proxy trading cepat: setiap hari beli top-k skor tertinggi, hold horizon, return setelah biaya."""
    d = df[[score_col, ret_col, "date"]].dropna()
    top = d.sort_values(score_col, ascending=False).groupby("date").head(k)
    net = top[ret_col] - cost
    daily = net.groupby(top["date"]).mean()
    return {"topk_mean_net_return": float(net.mean()), "topk_hit_rate": float((net > 0).mean()),
            "topk_daily_sharpe": float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0,
            "universe_mean_return": float(d[ret_col].mean() - cost)}


def feature_ic_report(df: pd.DataFrame, features: list[str], target: str, max_dates: int = 400) -> pd.DataFrame:
    """Information coefficient (Spearman cross-sectional per tanggal) — hanya dari data training."""
    dates = np.sort(df["date"].unique())
    if len(dates) > max_dates:
        dates = dates[np.linspace(0, len(dates) - 1, max_dates).astype(int)]
    sub = df[df["date"].isin(dates)]
    rows = []
    ranked = sub.groupby("date")[features + [target]].rank(pct=True)
    ranked["date"] = sub["date"]
    import warnings
    for f in features:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ics = ranked.groupby("date").apply(lambda g: g[f].corr(g[target]), include_groups=False).dropna()
        rows.append({"feature": f, "ic_mean": ics.mean(), "ic_std": ics.std(),
                     "ic_ir": ics.mean() / ics.std() if ics.std() > 0 else np.nan, "coverage": float(sub[f].notna().mean())})
    return pd.DataFrame(rows).sort_values("ic_mean", key=np.abs, ascending=False).reset_index(drop=True)
