"""Retraining + model deployment policy (§15, §16).

New model → walk-forward validation → bandingkan dengan model ACTIVE → promote HANYA bila lebih baik & stabil.
Bila tidak → REJECTED dan model lama tetap aktif.

Gerbang absolut (config/model.yaml → promotion):
  AUC rata-rata >= min_auc, std AUC <= max_auc_std, ECE <= max_ece, proxy trading positif di >= min_positive_fold_ratio fold.
Head-to-head (bila ada model ACTIVE dan cukup data baru setelah train_end-nya):
  holdout = data SETELAH model aktif dilatih → tidak pernah dilihat model aktif.
  Kandidat versi research dilatih s/d awal holdout − embargo → juga tidak melihat holdout.
  Kandidat ditolak bila log loss holdout > log loss model aktif + max_logloss_degradation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app import FEATURE_VERSION
from app.backtest.walk_forward import make_folds
from app.config import config_hash, embargo_days, get
from app.models.evaluation import classification_metrics, expected_calibration_error
from app.models.trainer import _train_rows, compare_models, fit_swing_model, walk_forward_predict
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


def _oos_metrics(pred: pd.DataFrame) -> dict:
    m = pred["y_class"].notna() & pred["is_tradeable"]
    p = pred[m]
    if p.empty:
        return {}
    P = p[["probability_bearish", "probability_neutral", "probability_bullish"]].to_numpy()
    y = p["y_class"].astype(int).to_numpy()
    out = classification_metrics(y, P)
    out["ece_bullish"] = expected_calibration_error((y == 2).astype(int), P[:, 2])
    return out


def _holdout_logloss(model, df: pd.DataFrame, mask: np.ndarray) -> float | None:
    rows = df[mask & df["is_tradeable"].to_numpy() & df["y_class"].notna().to_numpy()]
    if len(rows) < 200:
        return None
    try:
        p = model.predict(rows)
    except Exception as e:  # fitur berubah (feature_version beda) → tidak bisa dibandingkan langsung
        log.warning("Model aktif tidak bisa dievaluasi di holdout: %s", e)
        return None
    return classification_metrics(rows["y_class"].astype(int).to_numpy(),
                                  p[["probability_bearish", "probability_neutral", "probability_bullish"]].to_numpy())["log_loss"]


def evaluate_gates(metrics: dict, cfg: dict) -> list[str]:
    pc = cfg.get("promotion", {})
    fails = []
    if not metrics.get("auc_mean", 0) >= pc.get("min_auc", 0.52):
        fails.append(f"AUC rata-rata {metrics.get('auc_mean', float('nan')):.3f} < {pc.get('min_auc')}")
    if not metrics.get("auc_std", 1) <= pc.get("max_auc_std", 0.05):
        fails.append(f"std AUC {metrics.get('auc_std', float('nan')):.3f} > {pc.get('max_auc_std')} (tidak stabil)")
    if not metrics.get("ece_bullish", 1) <= pc.get("max_ece", 0.06):
        fails.append(f"ECE {metrics.get('ece_bullish', float('nan')):.3f} > {pc.get('max_ece')} (kalibrasi buruk)")
    if not metrics.get("positive_fold_ratio", 0) >= pc.get("min_positive_fold_ratio", 0.5):
        fails.append(f"proxy trading positif hanya di {metrics.get('positive_fold_ratio', 0):.0%} fold")
    return fails


def retrain(cfg: dict, df: pd.DataFrame, storage, force: bool = False, compare: bool = True) -> dict:
    t0 = time.time()
    active = storage.active_version()
    if active and not force:
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(active["promoted_at"] or active["created_at"])).days
        if age < int(get(cfg, "model.retrain_frequency_days", 30)):
            msg = f"Model aktif {active['version']} baru berumur {age} hari (< retrain_frequency_days) — retrain dilewati"
            log.info(msg)
            return {"decision": "SKIPPED", "reason": msg, "active_version": active["version"]}

    if compare or get(cfg, "model.classifier", "auto") == "auto":
        comp = compare_models(df, cfg)
        names, feats = comp["selected"], comp["features"]
        ranking = comp["ranking"]
    else:
        from app.models.trainer import select_features
        names = [get(cfg, "model.classifier")]
        folds = make_folds(df["date"], cfg)
        feats = select_features(df, (df["date"] <= folds[0].train_end).to_numpy(), cfg)[0]
        ranking = []

    oos_val, fm = walk_forward_predict(df, cfg, names, kinds=("validation",), features=feats)
    oos_test, fm_test = walk_forward_predict(df, cfg, names, kinds=("test",), features=feats)
    vm = _oos_metrics(oos_val)
    metrics = {
        "classifiers": names, "n_features": len(feats),
        "auc_mean": float(fm["roc_auc_bullish"].mean()), "auc_std": float(fm["roc_auc_bullish"].std(ddof=0)),
        "logloss_mean": float(fm["log_loss"].mean()), "brier_mean": float(fm["brier_bullish"].mean()),
        "ece_bullish": vm.get("ece_bullish"), "positive_fold_ratio": float((fm["topk_mean_net_return"] > 0).mean()),
        "topk_net_mean": float(fm["topk_mean_net_return"].mean()),
        "folds": fm[["name", "eval_start", "eval_end", "roc_auc_bullish", "log_loss", "brier_bullish",
                     "topk_mean_net_return"]].to_dict(orient="records"),
        "test_sealed": fm_test[["eval_start", "eval_end", "roc_auc_bullish", "log_loss", "topk_mean_net_return"]]
        .to_dict(orient="records") if len(fm_test) else [],
        "model_comparison": ranking,
    }

    # ---- head-to-head vs model aktif di data yang belum pernah dilihat model aktif
    h2h = None
    if active:
        holdout_start = pd.Timestamp(active["train_end"]) + pd.Timedelta(days=1) if active.get("train_end") else None
        if holdout_start is not None:
            ud = np.sort(df["date"].unique())
            hmask = (df["date"] >= holdout_start).to_numpy()
            n_days = int((ud >= np.datetime64(holdout_start)).sum())
            if n_days >= int(get(cfg, "promotion.holdout_min_trading_days", 40)):
                emb = embargo_days(cfg)
                pos = np.searchsorted(ud, np.datetime64(holdout_start))
                research_end = pd.Timestamp(ud[max(0, pos - emb - 1)])
                tr = _train_rows(df, (df["date"] <= research_end).to_numpy() & df["y_class"].notna().to_numpy(), cfg)
                research = fit_swing_model(tr, feats, cfg, names, with_aux=False)
                active_model = storage.load(active["version"])
                ll_new, ll_old = _holdout_logloss(research, df, hmask), _holdout_logloss(active_model, df, hmask)
                h2h = {"holdout_start": str(holdout_start.date()), "holdout_trading_days": n_days,
                       "candidate_logloss": ll_new, "active_logloss": ll_old}
    metrics["head_to_head"] = h2h

    # ---- model production: semua data berlabel
    labeled = df["y_class"].notna().to_numpy()
    prod = fit_swing_model(_train_rows(df, labeled, cfg), feats, cfg, names)
    meta = {"train_start": prod.meta["train_start"], "train_end": prod.meta["train_end"], "feature_version": FEATURE_VERSION,
            "config_hash": config_hash(cfg), "classifiers": names, "metrics": metrics}
    prod.meta.update({"feature_version": FEATURE_VERSION, "config_hash": meta["config_hash"]})

    fails = evaluate_gates(metrics, cfg)
    if h2h and h2h["candidate_logloss"] is not None and h2h["active_logloss"] is not None:
        tol = float(get(cfg, "promotion.max_logloss_degradation", 0.002))
        if h2h["candidate_logloss"] > h2h["active_logloss"] + tol:
            fails.append(f"holdout log loss {h2h['candidate_logloss']:.4f} lebih buruk dari model aktif {h2h['active_logloss']:.4f}")
    version = storage.save(prod, meta, status="CANDIDATE")
    if fails:
        storage.reject(version, "; ".join(fails))
        decision = "REJECTED"
    else:
        storage.promote(version, "gates passed" + (" + beat active on holdout" if h2h else ""))
        decision = "PROMOTED"
    out = {"decision": decision, "version": version, "previous_active": active["version"] if active else None,
           "reasons": fails, "metrics": metrics, "seconds": round(time.time() - t0, 1)}
    log.info("Retrain selesai: %s %s (%s) dalam %.0fs", version, decision, "; ".join(fails) or "semua gerbang lolos", out["seconds"])
    return out
