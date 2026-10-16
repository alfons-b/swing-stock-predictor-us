"""Training primitives (dipakai oleh app.models.retrain). Asal: pipeline research yang sudah teruji.

Training pipeline: dataset → model comparison (walk-forward) → OOS predictions → model final.

Alur (dipanggil dari app.models.retrain):
1. `compare_models`      : setiap kandidat dievaluasi di fold VALIDASI saja → skor gabungan
                           (prediktif + trading + kalibrasi + stabilitas).
2. `walk_forward_predict`: prediksi out-of-sample per fold (dipakai backtest & gerbang promosi).
3. `fit_swing_model`     : melatih ensemble + regressor + model hit-TP.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.backtest.walk_forward import fold_masks, make_folds
from app.config import artifacts_dir, embargo_days, get
from app.features.builder import feature_columns
from app.models.calibration import fit_calibrated
from app.models.ensemble import SwingModel
from app.models.evaluation import (classification_metrics, feature_ic_report, regression_metrics,
                                        topk_trading_proxy)
from app.models.factory import (available_models, make_binary_classifier, make_classifier, make_regressor,
                                     regressor_family_for)
from app.utils.io import save_json
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def round_trip_cost(cfg: dict) -> float:
    b = cfg["backtest"]
    return b["TRANSACTION_FEE_BUY"] + b["TRANSACTION_FEE_SELL"] + 2 * b["SLIPPAGE_BPS"] / 1e4


def _train_rows(df: pd.DataFrame, mask: np.ndarray, cfg: dict) -> pd.DataFrame:
    sub = df[mask & df["is_tradeable"].to_numpy() & df["y_class"].notna().to_numpy()]
    cap = get(cfg, "model.max_train_rows", 400000)
    if len(sub) > cap:
        sub = sub.sample(cap, random_state=get(cfg, "project.seed", 42)).sort_values("date")
    return sub


def fit_swing_model(train: pd.DataFrame, features: list[str], cfg: dict, classifier_names: list[str],
                    with_aux: bool = True) -> SwingModel:
    seed = get(cfg, "project.seed", 42)
    h = cfg["labels"]["SWING_HORIZON"]
    X = train[features].astype(float)
    y = train["y_class"].astype(int).to_numpy()
    dates = train["date"]
    method, frac, emb = get(cfg, "model.calibration", "sigmoid"), get(cfg, "model.calibration_fraction", 0.15), embargo_days(cfg)
    model = SwingModel(features=features, horizon=h, horizons=list(cfg["labels"]["HORIZONS"]))
    for name in classifier_names:
        model.classifiers[name] = fit_calibrated(make_classifier(name, seed), X, y, dates, method, frac, emb)
    if with_aux:
        reg_name = get(cfg, "model.regressor", "auto")
        if reg_name == "auto":
            reg_name = regressor_family_for(classifier_names[0])
        for hh in cfg["labels"]["HORIZONS"]:
            m = train[f"y_ret_{hh}d"].notna().to_numpy()
            model.regressors[f"return_{hh}d"] = make_regressor(reg_name, seed).fit(X[m], train.loc[m, f"y_ret_{hh}d"])
        for tgt in ("mfe", "mae"):
            m = train[f"y_{tgt}_{h}d"].notna().to_numpy()
            model.regressors[f"{tgt}_{h}d"] = make_regressor(reg_name, seed).fit(X[m], train.loc[m, f"y_{tgt}_{h}d"])
        for attr, col in (("hit_tp_model", "y_hit_tp"), ("hit_sl_model", "y_hit_sl_first")):
            m = train[col].notna().to_numpy()
            yy = train.loc[m, col].astype(int).to_numpy()
            if len(np.unique(yy)) == 2:
                setattr(model, attr, fit_calibrated(make_binary_classifier(classifier_names[0], seed), X[m], yy,
                                                    dates[m], method, frac, emb))
        model.meta["regressor_family"] = reg_name
    model.meta["feature_medians"] = X.median().to_dict()
    model.meta["base_rate_bullish"] = float((y == 2).mean())
    model.meta.update({"classifiers": classifier_names, "train_start": str(dates.min().date()),
                       "train_end": str(dates.max().date()), "n_train": int(len(train))})
    return model


def select_features(df: pd.DataFrame, train_mask: np.ndarray, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    feats = feature_columns(df, cfg)
    h = cfg["labels"]["SWING_HORIZON"]
    rows = _train_rows(df, train_mask, cfg)
    rep = feature_ic_report(rows, feats, f"y_ret_{h}d")
    thr = get(cfg, "model.feature_ic_min_abs", 0.0) or 0.0
    keep = rep.loc[(rep["ic_mean"].abs() >= thr) & (rep["coverage"] > 0.5), "feature"].tolist() if thr > 0 else \
        rep.loc[rep["coverage"] > 0.5, "feature"].tolist()
    return sorted(keep), rep


def evaluate_fold_predictions(pred: pd.DataFrame, cfg: dict) -> dict:
    h = cfg["labels"]["SWING_HORIZON"]
    m = pred["y_class"].notna() & pred["is_tradeable"]
    p = pred[m]
    P = p[["probability_bearish", "probability_neutral", "probability_bullish"]].to_numpy()
    out = classification_metrics(p["y_class"].astype(int).to_numpy(), P)
    if f"expected_return_{h}d" in p:
        out.update({f"reg_{k}": v for k, v in regression_metrics(p[f"y_ret_{h}d"].to_numpy(),
                                                                  p[f"expected_return_{h}d"].to_numpy()).items()})
    out.update(topk_trading_proxy(p, "probability_bullish", f"y_ret_{h}d", get(cfg, "model.comparison_top_k", 5),
                                  round_trip_cost(cfg)))
    return out


KEEP_COLS = ["date", "ticker", "is_tradeable", "y_class", "y_hit_tp"]


def walk_forward_predict(df: pd.DataFrame, cfg: dict, classifier_names: list[str], kinds=("validation",),
                         with_aux: bool = True, features: list[str] | None = None, folds=None):
    folds = folds if folds is not None else [f for f in make_folds(df["date"], cfg) if f.kind in kinds]
    preds, fold_rows = [], []
    for fold in folds:
        t = time.time()
        tr_m, ev_m = fold_masks(df, fold)
        feats = features or select_features(df, tr_m, cfg)[0]
        train = _train_rows(df, tr_m, cfg)
        model = fit_swing_model(train, feats, cfg, classifier_names, with_aux=with_aux)
        ev = df[ev_m]
        p = model.predict(ev)
        p = pd.concat([ev[KEEP_COLS + [c for c in ev.columns if c.startswith("y_ret") or c.startswith("y_m")]], p], axis=1)
        p["fold"] = fold.name
        preds.append(p)
        metrics = evaluate_fold_predictions(p, cfg)
        fold_rows.append({**fold.as_dict(), **metrics, "n_train": len(train), "seconds": round(time.time() - t, 1)})
        log.info("[%s] %s | AUC=%.3f logloss=%.3f topk_net=%.4f (%.0fs)", "+".join(classifier_names), fold.name,
                 metrics["roc_auc_bullish"], metrics["log_loss"], metrics["topk_mean_net_return"], time.time() - t)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), pd.DataFrame(fold_rows)


def _rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=s.index)
    r = s.rank(ascending=higher_better, pct=True)
    return (r - r.min()) / (r.max() - r.min())


def compare_models(df: pd.DataFrame, cfg: dict) -> dict:
    """AD. Bandingkan kandidat HANYA pada fold validasi. Test set tidak disentuh."""
    cands = available_models(get(cfg, "model.candidates", ["logistic", "hist_gbm"]))
    from app.backtest.walk_forward import assert_feasible
    folds = [f for f in assert_feasible(df["date"], cfg) if f.kind == "validation"]
    feats, ic_rep = select_features(df, fold_masks(df, folds[0])[0], cfg)
    rows, per_fold = [], []
    for name in cands:
        _, fm = walk_forward_predict(df, cfg, [name], with_aux=False, features=feats)
        fm["model"] = name
        per_fold.append(fm)
        rows.append({"model": name, "auc_mean": fm["roc_auc_bullish"].mean(), "auc_std": fm["roc_auc_bullish"].std(ddof=0),
                     "log_loss": fm["log_loss"].mean(), "brier": fm["brier_bullish"].mean(),
                     "pr_auc": fm["pr_auc_bullish"].mean(), "trading_mean": fm["topk_mean_net_return"].mean(),
                     "trading_std": fm["topk_mean_net_return"].std(ddof=0), "f1_macro": fm["f1_macro"].mean()})
    t = pd.DataFrame(rows)
    w = get(cfg, "model.comparison_weights")
    t["score_predictive"] = (_rank01(t["auc_mean"]) + _rank01(t["log_loss"], False) + _rank01(t["pr_auc"])) / 3
    t["score_trading"] = _rank01(t["trading_mean"])
    t["score_calibration"] = _rank01(t["brier"], False)
    t["score_stability"] = (_rank01(t["auc_std"], False) + _rank01(t["trading_std"], False)) / 2
    t["composite"] = sum(w[k] * t[f"score_{k}"] for k in ("predictive", "trading", "calibration", "stability"))
    t = t.sort_values("composite", ascending=False).reset_index(drop=True)
    n_ens = max(1, int(get(cfg, "model.ensemble_size", 1)))
    result = {"ranking": t.to_dict(orient="records"), "selected": t["model"].head(n_ens).tolist(),
              "features": feats, "per_fold": pd.concat(per_fold).to_dict(orient="records")}
    adir = artifacts_dir(cfg)
    save_json(result, adir / "model_comparison.json")
    ic_rep.to_csv(adir / "feature_ic_report.csv", index=False)
    log.info("Model comparison:\n%s", t[["model", "auc_mean", "log_loss", "brier", "trading_mean", "composite"]].to_string())
    return result


