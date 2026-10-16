"""Orkestrasi model challenger: prepare → train (per model) → predict harian.

Alur pelatihan (sama untuk semua model; tidak pernah menyentuh model aktif / tabel model_versions):
  A. seleksi  : latih s/d fold validasi terakhir − embargo, early stopping di fold itu   (LSTM/GRU/TFT)
  B. test     : latih s/d awal fold test − embargo (jumlah epoch dari A), prakirakan fold test → metrik tersegel
  C. produksi : latih s/d label terakhir yang tersedia → artifact untuk prakiraan harian
ARIMA-GARCH: B = estimasi parameter per emiten dari data ≤ akhir training test, parameter TETAP selama periode test;
C = estimasi ulang pada `window` hari terakhir.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from app.challengers import MODEL_LABELS, SEQUENCE_MODELS, challenger_cfg
from app.challengers import data as cd
from app.challengers.dist import normal_quantiles, summarize_quantiles
from app.challengers.calibration import calibrate_forecasts, fit_calibrators
from app.challengers.evaluate import (baseline_metrics, choose_ranking, compare_to_value, evaluate_forecasts, rank_score,
                                      significance, to_eval_frame, verdict)
from app.models.evaluation import topk_trading_proxy
from app.models.trainer import round_trip_cost
from app.config import get
from app.database.repository import now_utc
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

KEY_COLS = ["date", "ticker", "sector", "close", "volume", "is_tradeable"]


# ============================================================================== dataset (sekali per workflow)
def prepare_dataset(cfg: dict, repo, lookback: int | None = None, labels: bool = True) -> pd.DataFrame:
    """Dataset yang SAMA dengan model aktif (build_dataset) dipersempit ke kolom yang dibutuhkan challenger."""
    from app.pipeline.market_data import build_dataset
    df = build_dataset(cfg, repo, lookback, labels=labels)
    h = int(get(cfg, "labels.SWING_HORIZON", 5))
    keep = KEY_COLS + [c for c in df.columns if c.startswith("f_")] + [c for c in (f"y_ret_{h}d", "y_class") if c in df]
    return df[[c for c in keep if c in df.columns]].copy()


def save_dataset_npz(df: pd.DataFrame, path: str) -> None:
    """Simpan tanpa pickle (np.savez + allow_pickle=False saat dimuat)."""
    num = [c for c in df.columns if c not in ("date", "ticker", "sector")]
    np.savez_compressed(path, date=df["date"].to_numpy("datetime64[ns]").astype("int64"),
                        ticker=df["ticker"].astype(str).to_numpy(dtype="U12"),
                        sector=df["sector"].fillna("Unknown").astype(str).to_numpy(dtype="U64"),
                        values=df[num].to_numpy(dtype=np.float32, na_value=np.nan), columns=np.array(num, dtype="U64"))


def load_dataset_npz(path: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame(z["values"], columns=[str(c) for c in z["columns"]])
    df.insert(0, "sector", z["sector"].astype(object))
    df.insert(0, "ticker", z["ticker"].astype(object))
    df.insert(0, "date", pd.to_datetime(z["date"]))
    if "is_tradeable" in df:
        df["is_tradeable"] = df["is_tradeable"].fillna(0).astype(bool)
    return df


# ============================================================================== pembanding
def active_model_reference(repo) -> dict | None:
    """Metrik test tersegel model AKTIF (dibaca saja) untuk pembanding."""
    try:
        r = repo.db.query_df("SELECT version, metrics, train_end FROM model_versions WHERE status = ? "
                             "ORDER BY promoted_at DESC LIMIT 1", ("ACTIVE",))
    except Exception:
        return None
    if r.empty:
        return None
    m = json.loads(r["metrics"].iloc[0]) if isinstance(r["metrics"].iloc[0], str) else (r["metrics"].iloc[0] or {})
    ts = (m.get("test_sealed") or [{}])[0] if isinstance(m, dict) else {}
    return {"version": r["version"].iloc[0], "roc_auc_bullish": ts.get("roc_auc_bullish"), "log_loss": ts.get("log_loss"),
            "topk_mean_net_return": ts.get("topk_mean_net_return"), "eval_start": ts.get("eval_start"),
            "eval_end": ts.get("eval_end")}


def _baseline_values(panel, cc) -> np.ndarray | None:
    for name in (cc["baseline_feature"], "c_ret5"):
        if name in panel.features:
            return panel.X[:, panel.features.index(name)].astype(float)
    return None


def _validation_layer(panel, rows, q_log, cc, cfg) -> dict:
    """Kalibrator & aturan ranking dari prakiraan FOLD VALIDASI (test tidak dilihat)."""
    fc = summarize_quantiles(q_log, cc["quantiles"], cc["bull_threshold"], cc["bear_threshold"])
    ev = to_eval_frame(panel, rows, fc)
    mask = (ev["y_class"].notna() & ev["is_tradeable"].astype(bool)).to_numpy()
    calib = fit_calibrators(fc, ev["y_ret"].to_numpy(), cc["bull_threshold"], cc["bear_threshold"], mask)
    rule, scores = choose_ranking(to_eval_frame(panel, rows, calibrate_forecasts(fc, calib)), cfg)
    log.info("Lapisan validasi: kalibrasi %s, aturan ranking %s (top-k net validasi %s)", sorted(calib), rule,
             {k: round(v, 4) for k, v in scores.items()})
    return {"calibrators": calib, "ranking_rule": rule, "ranking_validation": scores, "n_validation": int(mask.sum())}


def _evaluate(panel, rows, q_log, cc, cfg, repo, layer: dict | None = None) -> dict:
    layer = layer or {}
    fc_raw = summarize_quantiles(q_log, cc["quantiles"], cc["bull_threshold"], cc["bear_threshold"])
    fc = calibrate_forecasts(fc_raw, layer.get("calibrators"))
    ev = to_eval_frame(panel, rows, fc)
    m = evaluate_forecasts(ev, cfg)
    raw = evaluate_forecasts(to_eval_frame(panel, rows, fc_raw), cfg)
    m["ece_bullish_raw"], m["log_loss_raw"] = raw.get("ece_bullish"), raw.get("log_loss")
    rule = layer.get("ranking_rule", "prob_bullish")
    ev["rank_score"] = rank_score(ev, rule)
    p = ev[ev["y_class"].notna() & ev["is_tradeable"].astype(bool)]
    m["ranking_rule"] = rule
    if len(p):
        m["topk_by_rule_net"] = topk_trading_proxy(p.assign(y_ret_h=p["y_ret"]), "rank_score", "y_ret_h",
                                                   int(get(cfg, "model.comparison_top_k", 5)), round_trip_cost(cfg))["topk_mean_net_return"]
    base_vals = _baseline_values(panel, cc)
    bv = base_vals[rows] if base_vals is not None else None
    base = baseline_metrics(ev, cfg, bv)
    sig = significance(ev, cfg, "rank_score", bv)                       # aturan terpilih (utama)
    sig_pb = significance(ev.assign(_pb=ev["prob_bullish"]), cfg, "_pb")   # definisi model aktif (untuk pembanding)
    if sig_pb:
        sig["topk_prob_bullish"] = sig_pb["topk"]
    ref = active_model_reference(repo)
    if ref:
        ref["comparison"] = compare_to_value(sig.get("topk_prob_bullish"), ref.get("topk_mean_net_return"))
    return {"test": m, "baselines": base, "active_model": ref, "significance": sig, "validation_layer": layer,
            "verdict": verdict(m, base, ref, sig)}


# ============================================================================== training
def train_sequence(model: str, df: pd.DataFrame, cfg: dict, repo) -> tuple[dict, dict, bytes, dict]:
    from app.challengers import trainer as T
    cc = challenger_cfg(cfg, model)
    sq, tr, h, seed = cc["sequence"], cc["training"], cc["horizon"], cc["seed"]
    L = int(sq.get("seq_len", 60))
    plan = cd.split_plan(df["date"], cfg)
    sel_mask = (df["date"] <= plan.select_train_end).to_numpy()
    feats = cd.select_sequence_features(df, sel_mask, int(sq.get("max_features", 64)),
                                        float(sq.get("min_feature_coverage", 0.6)), cfg)
    panel = cd.prepare_panel(df, feats, h, cc["target_clip"])
    min_hist = cc["min_history_days"]

    def stage_spec(train_end):
        rows = np.flatnonzero(panel.dates <= np.datetime64(pd.Timestamp(train_end)))
        center, scale = cd.fit_scaler(panel.X, rows)
        Z = cd.apply_scaler(panel.X, center, scale, float(sq.get("clip_z", 5.0)))
        ends = cd.eligible_ends(panel, L, plan.train_start, train_end, min_history=min_hist)
        ends = T.finite_target_ends(model, panel, ends)
        t = T.targets_for(model, panel, ends)
        y_mu, y_sd = (t.mean(axis=0), t.std(axis=0) + 1e-6)
        spec = T.SequenceSpec(model, feats, panel.sectors, L, h, cc["quantiles"], cc["arch"], center, scale, y_mu, y_sd,
                              float(sq.get("clip_z", 5.0)))
        return spec, Z, ends

    timings = {}
    # A. seleksi epoch
    t0 = time.time()
    spec_a, Z, tr_a = stage_spec(plan.select_train_end)
    val = T.finite_target_ends(model, panel, cd.eligible_ends(panel, L, *plan.select_eval, min_history=min_hist))
    model_a, info_a = T.fit_sequence_model(spec_a, panel, Z, tr_a, tr, seed, val_ends=val)
    best_epochs = max(int(tr.get("min_epochs", 2)), info_a["best_epoch"])
    val_rows = T.finite_target_ends(model, panel, cd.eligible_ends(panel, L, *plan.select_eval, min_history=min_hist))
    layer = _validation_layer(panel, val_rows, T.predict_quantiles(model_a, spec_a, panel, Z, val_rows), cc, cfg)
    timings["select_s"] = round(time.time() - t0, 1)
    # B. evaluasi tersegel pada fold test
    t0 = time.time()
    if tr.get("refit_for_test", True):
        spec_b, Z, tr_b = stage_spec(plan.test_train_end)
        model_b, info_b = T.fit_sequence_model(spec_b, panel, Z, tr_b, tr, seed + 1, epochs=best_epochs)
    else:
        spec_b, model_b, info_b = spec_a, model_a, info_a
        Z = cd.apply_scaler(panel.X, spec_a.center, spec_a.scale, spec_a.clip_z)
    test_rows = cd.eligible_ends(panel, L, *plan.test_eval, min_history=min_hist)
    q_log = T.predict_quantiles(model_b, spec_b, panel, Z, test_rows)
    evaluation = _evaluate(panel, test_rows, q_log, cc, cfg, repo, layer)
    timings["test_s"] = round(time.time() - t0, 1)
    # C. produksi
    t0 = time.time()
    last_label = pd.Timestamp(panel.dates[np.isfinite(panel.y)].max())
    if tr.get("production_refit", True):
        spec_c, Z, tr_c = stage_spec(last_label)
        model_c, info_c = T.fit_sequence_model(spec_c, panel, Z, tr_c, tr, seed + 2, epochs=best_epochs)
    else:
        spec_c, model_c, info_c, last_label = spec_b, model_b, info_b, plan.test_train_end
    timings["production_s"] = round(time.time() - t0, 1)
    artifact = T.save_artifact(model_c, spec_c)
    params = {"features": feats, "seq_len": L, "horizon": h, "quantiles": cc["quantiles"], "arch": cc["arch"],
              "training": tr, "n_parameters": int(sum(p.numel() for p in model_c.parameters())),
              "best_epochs": best_epochs, "artifact_bytes": len(artifact),
              "calibrators": layer["calibrators"], "ranking_rule": layer["ranking_rule"]}
    metrics = {**evaluation, "selection": {k: v for k, v in info_a.items()}, "test_fit": {k: v for k, v in info_b.items() if k != "history"},
               "production_fit": {k: v for k, v in info_c.items() if k != "history"}, "timings": timings}
    dates = {"train_start": plan.train_start, "train_end": last_label, "eval_start": plan.test_eval[0], "eval_end": plan.test_eval[1]}
    return params, metrics, artifact, dates


def _stock_blocks(panel):
    """Baris panel terurut per emiten → (kode emiten, awal, akhir) per blok kontigu (O(N), bukan O(N×S))."""
    codes, starts = np.unique(panel.stock, return_index=True)
    order = np.argsort(starts)
    starts = starts[order]
    ends = np.r_[starts[1:], len(panel.stock)]
    return list(zip(codes[order], starts, ends))


def _returns_by_stock(panel, upto=None) -> dict:
    out = {}
    lim = np.datetime64(pd.Timestamp(upto)) if upto is not None else None
    r_idx = panel.features.index("c_ret1")
    for _, a, b in _stock_blocks(panel):
        r = panel.X[a:b, r_idx].astype(float)
        if lim is not None:
            r = r[panel.dates[a:b] <= lim]
        if len(r):
            out[panel.tickers[a]] = r
    return out


def _arima_forecast_rows(panel, params: dict, rows: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """μ_H, σ_H untuk baris `rows` (parameter tetap per emiten). Emiten tanpa parameter → dilewati."""
    from app.challengers.arima_garch import forecast_series
    r_idx = panel.features.index("c_ret1")
    rows = np.sort(np.asarray(rows))
    keep, mus, sds = [], [], []
    for _, a, b in _stock_blocks(panel):
        lo, hi = np.searchsorted(rows, a), np.searchsorted(rows, b)
        if lo == hi:
            continue
        p = params.get(panel.tickers[a])
        if not p or p.get("status") != "OK":
            continue
        mu_h, sd_h = forecast_series(panel.X[a:b, r_idx], p, h)
        want = rows[lo:hi]
        keep.append(want)
        mus.append(mu_h[want - a])
        sds.append(sd_h[want - a])
    if not keep:
        return np.array([], int), np.array([]), np.array([])
    return np.concatenate(keep), np.concatenate(mus), np.concatenate(sds)


def train_arima_garch(df: pd.DataFrame, cfg: dict, repo) -> tuple[dict, dict, bytes, dict]:
    from app.challengers.arima_garch import fit_all, to_artifact
    cc = challenger_cfg(cfg, "arima_garch")
    a, h = cc["arch"], cc["horizon"]
    window, min_obs = int(a.get("window", 750)), int(a.get("min_obs", 250))
    orders = [tuple(o) for o in a.get("orders", [[0, 0], [1, 0], [0, 1], [1, 1]])]
    garch_cfg = a.get("garch", {"p": 1, "q": 1, "dist": "t"})
    n_jobs = int(a.get("n_jobs", -1))
    plan = cd.split_plan(df["date"], cfg)
    panel = cd.prepare_panel(df, ["c_ret1", "c_ret5"], h, cc["target_clip"])
    timings = {}
    # A. lapisan validasi: estimasi dari data ≤ akhir training validasi, parameter tetap selama fold validasi
    t0 = time.time()
    rets = {t: r[-window:] for t, r in _returns_by_stock(panel, plan.select_train_end).items()}
    params_val = fit_all(rets, orders, garch_cfg, min_obs, n_jobs)
    val_rows = cd.eligible_ends(panel, 1, *plan.select_eval, min_history=cc["min_history_days"])
    vrows, vmu, vsd = _arima_forecast_rows(panel, params_val, val_rows, h)
    layer = _validation_layer(panel, vrows, normal_quantiles(vmu, vsd, cc["quantiles"]), cc, cfg)
    timings["validation_s"] = round(time.time() - t0, 1)
    # B. evaluasi tersegel: estimasi dari data ≤ akhir training test, parameter tetap selama periode test
    t0 = time.time()
    rets = {t: r[-window:] for t, r in _returns_by_stock(panel, plan.test_train_end).items()}
    params_test = fit_all(rets, orders, garch_cfg, min_obs, n_jobs)
    test_rows = cd.eligible_ends(panel, 1, *plan.test_eval, min_history=cc["min_history_days"])
    rows, mu, sd = _arima_forecast_rows(panel, params_test, test_rows, h)
    q_log = normal_quantiles(mu, sd, cc["quantiles"])
    evaluation = _evaluate(panel, rows, q_log, cc, cfg, repo, layer)
    timings["test_s"] = round(time.time() - t0, 1)
    # C. produksi: estimasi ulang pada window terakhir
    t0 = time.time()
    rets = {t: r[-window:] for t, r in _returns_by_stock(panel).items()}
    params_prod = fit_all(rets, orders, garch_cfg, min_obs, n_jobs)
    timings["production_s"] = round(time.time() - t0, 1)
    ok = [p for p in params_prod.values() if p.get("status") == "OK"]
    order_counts = pd.Series([f"({p['p']},{p['q']})" for p in ok]).value_counts().to_dict() if ok else {}
    summary = {"stocks_fitted": len(ok), "stocks_insufficient": len(params_prod) - len(ok),
               "ewma_fallback": int(sum(p.get("fallback") == "ewma" for p in ok)), "arma_orders": order_counts,
               "median_persistence": float(np.median([p["alpha"] + p["beta"] for p in ok if "alpha" in p])) if ok else None}
    artifact = to_artifact(params_prod, {"horizon": h, "window": window, "fitted_at": now_utc()})
    params = {"window": window, "orders": orders, "garch": garch_cfg, "horizon": h, "quantiles": cc["quantiles"],
              "fit_summary": summary, "artifact_bytes": len(artifact),
              "calibrators": layer["calibrators"], "ranking_rule": layer["ranking_rule"]}
    metrics = {**evaluation, "timings": timings,
               "test_fit": {"stocks_fitted": sum(p.get("status") == "OK" for p in params_test.values())}}
    dates = {"train_start": plan.train_start, "train_end": pd.Timestamp(panel.dates.max()),
             "eval_start": plan.test_eval[0], "eval_end": plan.test_eval[1]}
    return params, metrics, artifact, dates


def run_train(model: str, df: pd.DataFrame, cfg: dict, repo, run_id: str) -> dict:
    t0 = time.time()
    fn = train_arima_garch if model == "arima_garch" else train_sequence
    params, metrics, artifact, dates = fn(model, df, cfg, repo) if model in SEQUENCE_MODELS else fn(df, cfg, repo)
    metrics["seconds_total"] = round(time.time() - t0, 1)
    version = repo.next_challenger_version(model)

    def _d(x):
        return None if x is None else str(pd.Timestamp(x).date())
    repo.save_challenger_model({"model_name": model, "version": version, "status": "READY", "created_at": now_utc(),
                                "train_start": _d(dates["train_start"]), "train_end": _d(dates["train_end"]),
                                "eval_start": _d(dates["eval_start"]), "eval_end": _d(dates["eval_end"]),
                                "params": params, "metrics": _clean(metrics), "artifact": artifact, "run_id": run_id,
                                "notes": "challenger — tidak memengaruhi keputusan BUY/WAIT"})
    pruned = repo.prune_challenger_artifacts(model, challenger_cfg(cfg)["artifacts_keep"])
    t = metrics["test"]
    log.info("Challenger %s %s: AUC bullish %s, top-k net %s, IC %s (%.0fs). %s", MODEL_LABELS[model], version,
             _fmt(t.get("roc_auc_bullish")), _fmt(t.get("topk_mean_net_return"), pct=True), _fmt(t.get("ic_mean")),
             metrics["seconds_total"], " | ".join(metrics["verdict"]), extra={"persist": True})
    return {"model": model, "version": version, "test": t, "baselines": metrics["baselines"],
            "significance": metrics.get("significance", {}),
            "active_model": metrics["active_model"], "verdict": metrics["verdict"], "seconds": metrics["seconds_total"],
            "artifacts_pruned": pruned}


# ============================================================================== prakiraan harian
def run_predict(models: list[str], df: pd.DataFrame, cfg: dict, repo, run_id: str) -> dict:
    """Prakiraan untuk tanggal terbaru di `df` (labels=False) dengan versi READY terbaru tiap model."""
    out = {}
    last = pd.Timestamp(df["date"].max())
    for model in models:
        latest = repo.latest_challenger(model)
        if not latest:
            out[model] = {"status": "NO_MODEL", "detail": "jalankan workflow Challenger Train"}
            continue
        cc = challenger_cfg(cfg, model)
        try:
            if model == "arima_garch":
                from app.challengers.arima_garch import from_artifact
                params, meta = from_artifact(latest["artifact"])
                h = int(meta.get("horizon", cc["horizon"]))
                panel = cd.prepare_panel(df, ["c_ret1", "c_ret5"], h, cc["target_clip"])
                rows = cd.latest_ends(panel, 1)                    # emiten tanpa parameter (histori pendek) dilewati
                rows, mu, sd = _arima_forecast_rows(panel, params, rows, h)
                q_log = normal_quantiles(mu, sd, cc["quantiles"])
                levels = cc["quantiles"]
            else:
                from app.challengers import trainer as T
                net, spec = T.load_artifact(latest["artifact"])
                h, levels = spec.horizon, spec.levels
                panel = cd.prepare_panel(df, spec.features, h, cc["target_clip"], sectors=spec.sectors)
                Z = cd.apply_scaler(panel.X, spec.center, spec.scale, spec.clip_z)
                rows = cd.latest_ends(panel, spec.seq_len)
                q_log = T.predict_quantiles(net, spec, panel, Z, rows)
            fc = summarize_quantiles(q_log, levels, cc["bull_threshold"], cc["bear_threshold"])
            mp = json.loads(latest["params"]) if isinstance(latest.get("params"), str) else (latest.get("params") or {})
            fc = calibrate_forecasts(fc, mp.get("calibrators"))
            rule = mp.get("ranking_rule", "prob_bullish")
            fc["rank_score"] = rank_score(fc, rule)
            frame = pd.DataFrame({"forecast_date": last.strftime("%Y-%m-%d"), "ticker": panel.tickers[rows],
                                  "model_name": model, "model_version": latest["version"], "horizon": h,
                                  **{k: fc[k] for k in ("exp_return", "prob_up", "prob_bullish", "prob_bearish", "vol",
                                                        "q10", "q50", "q90", "rank_score")},
                                  "run_id": run_id, "created_at": now_utc()})
            repo.save_model_forecasts(frame)
            out[model] = {"status": "OK", "version": latest["version"], "n": int(len(frame)), "date": str(last.date()),
                          "ranking_rule": rule, "calibrated": sorted(mp.get("calibrators") or {})}
            log.info("Prakiraan %s %s: %d emiten untuk %s", MODEL_LABELS[model], latest["version"], len(frame), last.date(),
                     extra={"persist": True})
        except Exception as e:                                          # satu model gagal tidak menghentikan yang lain
            log.warning("Prakiraan %s gagal: %s", model, e, exc_info=True, extra={"persist": True})
            out[model] = {"status": "FAILED", "detail": f"{type(e).__name__}: {e}"[:300]}
    return out


# ============================================================================== util
def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.2%}" if pct else f"{v:.3f}"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    return o
