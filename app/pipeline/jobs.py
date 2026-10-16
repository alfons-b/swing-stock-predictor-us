"""Orkestrasi job (§6, §11, §16, §17). Hanya mengorkestrasi — logika ada di layer masing-masing (§46).

Setiap job: START → PROCESS → SAVE → EXIT (§55). Tidak ada proses yang dibiarkan hidup.
"""
from __future__ import annotations

import json

import pandas as pd

from app import FEATURE_VERSION
from app.config import get
from app.data.ingestion import run_ingestion
from app.data.universe import UniverseManager
from app.data.validator import issues_to_frame, validate_prices
from app.pipeline.context import AppContext, PipelineRun
from app.pipeline.evaluate import evaluate_predictions
from app.pipeline.freshness import data_freshness
from app.pipeline.health import check_environment, health_check
from app.pipeline.market_data import build_dataset
from app.reporting.daily_report import html_report, markdown_report, predictions_csv, sector_table
from app.scanner.coverage import build_coverage, coverage_csv, coverage_summary
from app.scanner.scanner import analyze_ticker, scan_day, summarize_scan
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


class PipelineAbort(RuntimeError):
    pass


# ---------------------------------------------------------------------------------- helpers
def _ingest(ctx: AppContext, run: PipelineRun, full_actions: bool = False) -> dict:
    try:
        uni = UniverseManager(ctx.cfg, ctx.repo, ctx.chain).update()
    except Exception as e:  # universe gagal → lanjut dengan universe di DB (bila ada)
        if ctx.db.scalar("SELECT COUNT(*) FROM stocks") == 0:
            raise
        log.warning("Update universe gagal, memakai universe di database: %s", e)
        uni = {"error": str(e)}
    st = run_ingestion(ctx.cfg, ctx.repo, ctx.chain, ctx.calendar, full_actions=full_actions)
    run.stats.update({"stocks_processed": st.stocks_processed, "stocks_failed": st.stocks_failed,
                      "rows_inserted": st.rows_inserted, "rows_updated": st.rows_updated,
                      "market_date": st.expected_date})
    log.info("Ingestion: %d emiten diproses, %d gagal, %d up-to-date, +%d baris, %d diperbarui, %d ditolak validasi",
             st.stocks_processed, st.stocks_failed, st.stocks_up_to_date, st.rows_inserted, st.rows_updated,
             st.rows_rejected, extra={"persist": True})
    if st.failed:
        log.warning("Ticker gagal (%d): %s", len(st.failed), ", ".join(list(st.failed)[:30]))
    if st.stocks_requested and st.stocks_failed / st.stocks_requested > float(get(ctx.cfg, "pipeline.max_failed_ticker_ratio", 0.5)):
        raise PipelineAbort(f"{st.stocks_failed}/{st.stocks_requested} ticker gagal diunduh — provider bermasalah")
    if st.stocks_failed:
        run.status = "PARTIAL_SUCCESS"
    # daftar lengkap (as_dict memotong 50 untuk ringkasan) — dipakai status cakupan per emiten
    return {"universe": uni, "ingestion": st.as_dict(), "_failed_all": dict(st.failed), "_no_data": list(st.no_data)}


def _ensure_earnings_history(ctx: AppContext, min_share: float = 0.25) -> dict | None:
    """Fitur f2 memakai tanggal rilis laba SEC. Bila tabel earnings_events (hampir) kosong — upgrade pertama atau
    database baru — isi dulu SEBELUM model dilatih; kalau tidak, model belajar f_ear3 = NaN untuk semua baris lalu
    distribusinya bergeser begitu event mulai terisi. Gagal / SEC tidak tersedia → lanjut tanpa (dicatat di log)."""
    try:
        n_with = int(ctx.db.scalar("SELECT COUNT(DISTINCT stock_id) FROM earnings_events") or 0)
        n_active = int(ctx.db.scalar("SELECT COUNT(*) FROM stocks WHERE is_active = ?", (True,)) or 0)
    except Exception:
        return None
    if n_active == 0 or n_with >= min_share * n_active:
        return None
    from app.valuation.fundamental_provider import make_fundamental_providers, update_fundamentals
    if not any(hasattr(p, "get_earnings_events") for p in make_fundamental_providers(ctx.cfg)):
        return None
    log.info("Riwayat rilis laba baru tersedia untuk %d/%d saham — backfill dari SEC sebelum training", n_with, n_active,
             extra={"persist": True})
    try:
        st = update_fundamentals(ctx.cfg, ctx.repo, pd.Timestamp(ctx.cfg.get("_as_of") or pd.Timestamp.today()),
                                 earnings_backfill=True)
    except Exception as e:
        log.warning("Backfill earnings gagal (lanjut tanpa fitur earnings): %s", e, extra={"persist": True})
        return None
    bf = st.get("earnings_backfill") or {}
    return {"stocks_checked": int(st.get("checked", 0)) + int(bf.get("checked", 0)),
            "event_rows": int(st.get("earnings_events", 0)) + int(bf.get("rows", 0))}


def _model_needs_upgrade(ctx: AppContext) -> bool:
    try:
        _, active = ctx.models.load_active_model()
    except Exception:
        return False
    return active.get("feature_version") != FEATURE_VERSION


def _load_model(ctx: AppContext):
    model, active = ctx.models.load_active_model()
    if active.get("feature_version") != FEATURE_VERSION:
        if not get(ctx.cfg, "pipeline.auto_retrain_on_feature_change", True):
            raise PipelineAbort(f"Model {active['version']} memakai feature_version {active.get('feature_version')} "
                                f"≠ {FEATURE_VERSION}. Jalankan workflow retrain.")
        # Kode fitur/label baru ter-deploy → latih ulang SEKALI di sini agar daily tidak berhenti (±10–20 menit).
        log.warning("Model %s memakai feature_version %s ≠ %s — retrain otomatis sebelum scan",
                    active["version"], active.get("feature_version"), FEATURE_VERSION, extra={"persist": True})
        job_train(ctx, force=True)
        model, active = ctx.models.load_active_model()
        if active.get("feature_version") != FEATURE_VERSION:
            raise PipelineAbort("Retrain otomatis tidak menghasilkan model aktif dengan feature_version baru")
    return model, active["version"]


def _coverage(ctx: AppContext, df: pd.DataFrame, ingest: dict | None) -> pd.DataFrame:
    """Status cakupan setiap emiten aktif pada tanggal scan (dihitung SEBELUM ranking agar gerbang BUY berlaku)."""
    d = df["date"].max()
    ingest = ingest or {}
    return build_coverage(ctx.repo.stocks(), ctx.repo.latest_price_dates(), df[df["date"] == d], d, ctx.cfg,
                          ingest.get("_failed_all"), ingest.get("_no_data"))


def _record_coverage(ctx: AppContext, run: PipelineRun, cov: pd.DataFrame, summary: dict) -> dict:
    cs = coverage_summary(cov)
    summary["coverage"] = cs
    ctx.repo.save_coverage(cov, run.run_id)
    min_pct = float(get(ctx.cfg, "coverage.min_coverage_pct", 80))
    log.info("Cakupan scan %s: %d/%d dianalisis (%.1f%%), %d layak sinyal, status %s", cs["scan_date"], cs["processed"],
             cs["total_universe"], cs["coverage_pct"], cs["signal_eligible"], cs["status_counts"], extra={"persist": True})
    if cs["total_universe"] and cs["coverage_pct"] < min_pct:
        log.warning("Cakupan %.1f%% < minimum %.0f%% — run ditandai PARTIAL_SUCCESS", cs["coverage_pct"], min_pct,
                    extra={"persist": True})
        run.status = "PARTIAL_SUCCESS"
    return cs


def _driver_features(ctx, model) -> list[str]:
    ic = ctx.cfg.get("_ic_order") or []
    return [f for f in ic if f in model.features][:15] or model.features[:15]


def _save_scan(ctx: AppContext, res: dict, summary: dict, run_id: str, model_version: str) -> dict:
    cfg, repo = ctx.cfg, ctx.repo
    sig, ranked = res["signals"], res["ranked"]
    ids = repo.stock_ids()
    h = cfg["labels"]["SWING_HORIZON"]
    d = res["date"].strftime("%Y-%m-%d")
    rich = sig["decision"].isin(["BUY", "WATCHLIST"])
    from app.strategy.explain import build_reasoning
    reasons = [build_reasoning(r) if k else ([], []) for (_, r), k in zip(sig.iterrows(), rich)]
    pred = pd.DataFrame({
        "prediction_date": d, "stock_id": sig["ticker"].map(ids), "ticker": sig["ticker"], "model_version": model_version,
        "horizon": h, "decision": sig["decision"], "setup": sig["setup_type"], "score": sig["final_score"],
        "market_regime": sig["market_regime"], "prob_bearish": sig["probability_bearish"],
        "prob_neutral": sig["probability_neutral"], "prob_bullish": sig["probability_bullish"],
        "expected_return": sig.get(f"expected_return_{h}d"), "prob_hit_tp": sig.get("prob_hit_tp"),
        "entry_low": sig["entry_low"], "entry_ideal": sig["entry_ideal"], "entry_high": sig["entry_high"],
        "stop_loss": sig["stop_loss"], "tp1": sig["take_profit_1"], "tp2": sig["take_profit_2"],
        "risk_reward": sig["risk_reward"], "position_size": sig["position_size"], "lots": sig["position_lots"],
        "capital_required": sig["capital_required"], "estimated_loss": sig["estimated_loss"],
        "confidence": sig["confidence"], "close_price": sig["close"],
        "data_status": summary["data_freshness"]["status"],
        "reasons": [json.dumps(r[0], ensure_ascii=False) if k else None for r, k in zip(reasons, rich)],
        "risks": [json.dumps(r[1], ensure_ascii=False) if k else None for r, k in zip(reasons, rich)],
        "reject_reasons": sig["reject_reasons"].map(json.dumps), "run_id": run_id,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")})
    from app.research.pipeline import PRED_RESEARCH_COLS
    for k in PRED_RESEARCH_COLS:                       # skor riset terpisah (None bila modul/data tidak tersedia)
        if f"research_{k}" in sig:
            pred[k] = sig[f"research_{k}"].values
    pi, pu = repo.save_predictions(pred)
    sg = pd.DataFrame({"signal_date": d, "stock_id": ranked["ticker"].map(ids), "ticker": ranked["ticker"],
                       "rank": ranked["rank"], "decision": ranked["decision"], "setup": ranked["setup_type"],
                       "score": ranked["final_score"], "prob_bullish": ranked["probability_bullish"],
                       "expected_return": ranked.get(f"expected_return_{h}d"), "entry_low": ranked["entry_low"],
                       "entry_ideal": ranked["entry_ideal"], "entry_high": ranked["entry_high"],
                       "stop_loss": ranked["stop_loss"], "tp1": ranked["take_profit_1"], "tp2": ranked["take_profit_2"],
                       "risk_reward": ranked["risk_reward"], "lots": ranked["position_lots"],
                       "confidence": ranked["confidence"], "model_version": model_version, "run_id": run_id})
    repo.save_signals(sg, d)
    sec = sector_table(sig)
    repo.save_sector_data(sec.rename(columns={"score": "score"}).assign(date=d)[
        ["date", "sector", "ret5", "ret20", "ret60", "breadth", "volume_mom", "rs20", "score", "rank", "n_stocks"]])
    snap_cols = ["close", "ema20", "ema50", "sma200", "rsi14", "adx", "atr_pct", "volume_ratio", "f_rs20", "f_rs60",
                 "sector_score", "liquidity_score", "setup_type", "setup_score", "final_score", "technical_score", "ml_score"]
    feats = pd.DataFrame({"stock_id": sig["ticker"].map(ids), "date": d, "feature_version": FEATURE_VERSION,
                          "payload": sig[snap_cols].apply(lambda r: json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()},
                                                                               default=str), axis=1)})
    repo.save_features_snapshot(feats, int(get(cfg, "pipeline.features_snapshot_days", 5)))
    log.info("Tersimpan: %d prediksi (+%d baru), %d sinyal ranking, %d sektor", len(pred), pi, len(sg), len(sec),
             extra={"persist": True})
    return {"predictions": len(pred), "signals": len(sg)}


def _run_research(ctx: AppContext, df: pd.DataFrame, res: dict, run_id: str) -> dict:
    """Valuasi, foreign flow, akumulasi, ranking terintegrasi. Gagal → dicatat, pipeline swing tetap jalan."""
    from app.research.pipeline import attach_to_signals, run_research
    try:
        cols = [c for c in ("ticker", "date", "open", "high", "low", "close", "volume", "value") if c in df]
        prices = df[cols] if {"high", "low", "volume"} <= set(cols) else \
            ctx.repo.load_prices(start=(res["date"] - pd.Timedelta(days=400)).strftime("%Y-%m-%d"))
        out = run_research(ctx.cfg, ctx.repo, prices, res["signals"], res["date"], run_id)
        res["signals"] = attach_to_signals(res["signals"], out.get("table"))
        return out
    except Exception as e:
        log.warning("Modul riset gagal total (pipeline swing tetap jalan): %s", e, exc_info=True, extra={"persist": True})
        return {"summary": {"status": {"research": f"FAILED: {str(e)[:200]}"}}}


def _write_reports(ctx: AppContext, summary: dict, res: dict, df: pd.DataFrame, run_id: str, evaluation=None,
                   research_table: pd.DataFrame | None = None, coverage: pd.DataFrame | None = None) -> list[str]:
    ds = summary["date"].replace("-", "")
    sec = sector_table(res["signals"])
    files = [("predictions_csv", f"daily_predictions_{ds}.csv", predictions_csv(res["signals"], ctx.cfg["labels"]["SWING_HORIZON"])),
             ("html", f"daily_report_{ds}.html", html_report(summary, sec, df, evaluation).encode("utf-8")),
             ("markdown", f"daily_report_{ds}.md", markdown_report(summary, sec).encode("utf-8")),
             ("summary_json", f"daily_summary_{ds}.json", json.dumps(summary, default=str, ensure_ascii=False, indent=2).encode())]
    if coverage is not None and len(coverage):
        files.append(("coverage_csv", f"scan_coverage_{ds}.csv", coverage_csv(coverage)))
    if research_table is not None and len(research_table):
        from app.research.report import research_csv
        files.append(("research_csv", f"research_rankings_{ds}.csv", research_csv(research_table)))
    uris = [ctx.reports.save(summary["date"], kind, fn, content, run_id) for kind, fn, content in files]
    log.info("Report dibuat: %s", ", ".join(fn for _, fn, _ in files), extra={"persist": True})
    return uris


def _today(ctx: AppContext) -> pd.Timestamp:
    return pd.Timestamp(ctx.calendar.now().date())


def ensure_writable(ctx: AppContext) -> None:
    """Kuota storage hampir/sudah terlampaui → pulihkan dengan retensi + VACUUM FULL SEBELUM menulis.

    Supabase menandai database READ-ONLY; Neon/Aiven tidak memberi tanda apa pun, penulisan langsung gagal
    ("project size limit exceeded" / disk penuh). Karena itu pemicu kedua adalah rasio ukuran ≥ 95%."""
    from app.database.maintenance import run_maintenance, size_status
    sz = size_status(ctx.cfg, ctx.db)
    if not sz["read_only"] and sz["ratio"] < 0.95:
        return
    log.error("Database %s (%.0f MB, batas %.0f MB, %s) — menjalankan maintenance darurat",
              "READ-ONLY" if sz["read_only"] else "hampir penuh", sz["bytes"] / 1e6, sz["limit_bytes"] / 1e6,
              sz.get("provider"))
    res = run_maintenance(ctx.cfg, ctx.db, _today(ctx), full=True)
    if res["after_mb"] * 1e6 >= 0.95 * sz["limit_bytes"]:
        raise PipelineAbort(f"Database masih {res['after_mb']:.0f} MB setelah maintenance (batas {sz['limit_bytes'] / 1e6:.0f} MB). "
                            "Kurangi retention.price_history_years / predictions_nonsignal_days, atau naikkan paket "
                            "database lalu set Variable DB_SIZE_LIMIT_MB.")


def job_db_maintenance(ctx: AppContext, dry_run: bool = False, full: bool = False) -> dict:
    from app.database.maintenance import run_maintenance
    if ctx.db.dialect == "postgres":  # agar run tetap bisa dicatat walau database read-only
        ctx.db.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
    with PipelineRun(ctx, "db_maintenance") as run:
        out = run_maintenance(ctx.cfg, ctx.db, _today(ctx), dry_run=dry_run, full=full)
        run.stats["summary"] = {k: v for k, v in out.items() if k != "table_mb"}
        run.stats["rows_updated"] = sum(s["rows"] for s in out["steps"]) if not dry_run else 0
        return out


# ---------------------------------------------------------------------------------- jobs
def job_daily(ctx: AppContext, skip_ingest: bool = False) -> dict:
    ensure_writable(ctx)
    with PipelineRun(ctx, "daily") as run:
        h = health_check(ctx, check_provider=False)
        if not h["ok"]:
            raise PipelineAbort("Health check gagal: " + json.dumps({k: v for k, v in h["checks"].items() if v["status"] == "FAIL"}))
        ingest = {} if skip_ingest else _ingest(ctx, run)
        fresh = data_freshness(ctx.cfg, ctx.repo, ctx.calendar)
        run.stats["data_status"] = fresh["status"]
        if fresh["status"] == "UNAVAILABLE":
            raise PipelineAbort("Tidak ada data harga di database")
        if _model_needs_upgrade(ctx):
            run.stats["earnings_backfill"] = _ensure_earnings_history(ctx)
        df = build_dataset(ctx.cfg, ctx.repo, int(get(ctx.cfg, "pipeline.scan_lookback_trading_days", 520)), labels=False)
        model, version = _load_model(ctx)
        run.stats["model_version"] = version
        cov = _coverage(ctx, df, ingest)
        res = scan_day(ctx.cfg, df, model, fresh, coverage=cov)
        research = _run_research(ctx, df, res, run.run_id)
        summary = summarize_scan(ctx.cfg, res, model, version, fresh, ctx.is_synthetic(), _driver_features(ctx, model))
        summary["research"] = research.get("summary", {})
        _record_coverage(ctx, run, cov, summary)
        saved = _save_scan(ctx, res, summary, run.run_id, version)
        evaluation = evaluate_predictions(ctx.cfg, ctx.repo)
        uris = _write_reports(ctx, summary, res, df, run.run_id, evaluation, research.get("table"), cov)
        run.stats["summary"] = {"headline": summary["headline"], "date": summary["date"], "regime": summary["market_regime"],
                                "coverage": {k: v for k, v in summary["coverage"].items() if k != "top_reasons"},
                                "top": [r["ticker"] for r in summary["recommendations"]], "freshness": fresh, "saved": saved,
                                "evaluation": evaluation, "reports": uris, "ingestion": ingest.get("ingestion", {}).get("provider_stats")}
        try:
            from app.notifications.notify import notify_daily
            notify_daily(summary)
        except Exception as e:
            log.warning("Notifikasi gagal: %s", e)
        try:  # cegah kuota database terlampaui (paket gratis: penulisan diblokir di atas batas)
            from app.database.maintenance import run_maintenance, size_status
            sz = size_status(ctx.cfg, ctx.db)
            if sz["maintenance_due"]:
                log.warning("Database %.0f MB (%.0f%% batas) — retensi otomatis dijalankan", sz["bytes"] / 1e6, 100 * sz["ratio"])
                run.stats["summary"]["maintenance"] = run_maintenance(ctx.cfg, ctx.db, _today(ctx))["after_mb"]
        except Exception as e:
            log.warning("Maintenance otomatis gagal: %s", e)
        if run.status == "RUNNING":  # status final ditulis saat keluar dari `with`; nilai yang sama ditampilkan CLI
            run.status = "SUCCESS"
        summary["run_id"] = run.run_id
        summary["run_status"] = run.status
        return summary


def job_update_data(ctx: AppContext, full_actions: bool = False) -> dict:
    with PipelineRun(ctx, "update_data") as run:
        out = _ingest(ctx, run, full_actions=full_actions)
        run.stats["summary"] = out["ingestion"]
        return out


def job_validate(ctx: AppContext) -> pd.DataFrame:
    with PipelineRun(ctx, "validate_data") as run:
        prices = ctx.repo.load_prices()
        st = ctx.repo.stocks()
        uni = st[["ticker", "listing_date", "delisting_date"]].copy()
        ca = ctx.repo.load_actions()
        rep = issues_to_frame(validate_prices(prices, uni, ca, ctx.cfg))
        counts = ctx.repo.price_counts()
        rep = pd.concat([rep, pd.DataFrame([{"check": "db_duplicate_stock_date", "severity": "OK" if not counts["duplicates"] else "ERROR",
                                              "count": counts["duplicates"], "detail": "unique(stock_id, date)", "tickers": []}])])
        run.stats["summary"] = {"issues": rep[rep["severity"].isin(["WARN", "ERROR"])].to_dict(orient="records"), **counts}
        if (rep["severity"] == "ERROR").any():
            run.status = "PARTIAL_SUCCESS"
        return rep


def job_features(ctx: AppContext) -> dict:
    with PipelineRun(ctx, "features") as run:
        df = build_dataset(ctx.cfg, ctx.repo, int(get(ctx.cfg, "pipeline.scan_lookback_trading_days", 520)), labels=False)
        last = df[df["date"] == df["date"].max()]
        out = {"rows": len(df), "stocks": int(df["ticker"].nunique()), "date": str(df["date"].max().date()),
               "features": int(sum(c.startswith("f_") for c in df.columns)), "tradeable_today": int(last["is_tradeable"].sum()),
               "regime_today": last["market_regime"].iloc[0]}
        run.stats["summary"] = out
        return out


def job_train(ctx: AppContext, force: bool = True, compare: bool = True) -> dict:
    from app.models.retrain import retrain
    with PipelineRun(ctx, "retrain") as run:
        df = build_dataset(ctx.cfg, ctx.repo, None, labels=True)
        out = retrain(ctx.cfg, df, ctx.models, force=force, compare=compare)
        run.stats["model_version"] = out.get("version") or out.get("active_version")
        run.stats["summary"] = {k: v for k, v in out.items() if k != "metrics"}
        return out


def job_backtest(ctx: AppContext, last_n_folds: int | None = None, live: bool = True) -> dict:
    from app.backtest.runner import live_backtest, walk_forward_backtest
    with PipelineRun(ctx, "backtest") as run:
        model, version = _load_model(ctx)
        df = build_dataset(ctx.cfg, ctx.repo, None, labels=True)
        n = last_n_folds if last_n_folds is not None else get(ctx.cfg, "backtest.weekly_validation_folds")
        out = walk_forward_backtest(ctx.cfg, df, model.meta["classifiers"], model.features, ctx.repo, version, n)
        if live:
            out["live"] = live_backtest(ctx.cfg, ctx.repo, df)
        try:                                             # bukti per faktor (rank IC vs return residual) — dashboard
            from app.research.factor_report import run_factor_report
            fr = run_factor_report(ctx, df)
            out["factor_ic_top"] = fr.head(15).round(4).to_dict(orient="records") if len(fr) else []
        except Exception as e:  # laporan tambahan tidak boleh menggagalkan backtest
            log.warning("Laporan IC faktor gagal: %s", e)
        run.stats["model_version"] = version
        run.stats["summary"] = {k: {kk: vv for kk, vv in (v or {}).items() if kk in ("run_id", "strategy", "ihsg_buy_hold")}
                                for k, v in out.items() if isinstance(v, dict) or v is None}
        return out


def job_scan(ctx: AppContext, date: str | None = None, save: bool = True) -> dict:
    with PipelineRun(ctx, "scan") as run:
        fresh = data_freshness(ctx.cfg, ctx.repo, ctx.calendar)
        df = build_dataset(ctx.cfg, ctx.repo, int(get(ctx.cfg, "pipeline.scan_lookback_trading_days", 520)),
                           labels=False, end=date)
        model, version = _load_model(ctx)
        if date:  # scan historis: freshness tidak relevan, tetapi ditandai
            fresh = {**fresh, "status": "HISTORICAL", "trading_allowed": True, "latest_database_date": date}
        cov = None if date else _coverage(ctx, df, None)   # status historis tidak bisa direkonstruksi jujur
        res = scan_day(ctx.cfg, df, model, fresh, coverage=cov)
        summary = summarize_scan(ctx.cfg, res, model, version, fresh, ctx.is_synthetic(), _driver_features(ctx, model))
        if cov is not None:
            summary["coverage"] = coverage_summary(cov)
        if save and not date:
            _record_coverage(ctx, run, cov, summary)
            _save_scan(ctx, res, summary, run.run_id, version)
        run.stats.update({"model_version": version, "data_status": fresh["status"],
                          "summary": {"headline": summary["headline"], "date": summary["date"]}})
        return summary


def job_predict(ctx: AppContext, ticker: str, date: str | None = None) -> dict:
    fresh = data_freshness(ctx.cfg, ctx.repo, ctx.calendar)
    df = build_dataset(ctx.cfg, ctx.repo, int(get(ctx.cfg, "pipeline.scan_lookback_trading_days", 520)), labels=False, end=date)
    model, version = _load_model(ctx)
    cov = None if date else _coverage(ctx, df, None)
    res = scan_day(ctx.cfg, df, model, fresh, coverage=cov)
    rec = analyze_ticker(ctx.cfg, res, ticker, model, version, ctx.is_synthetic())
    rec["data_freshness"] = fresh["status"]
    if cov is not None:
        row = cov[cov["ticker"] == rec["ticker"]]
        rec["coverage_status"] = row["status"].iloc[0] if len(row) else "NOT_IN_UNIVERSE"
        rec["coverage_reason"] = row["reason"].iloc[0] if len(row) else ""
    return rec


def job_evaluate(ctx: AppContext) -> dict:
    with PipelineRun(ctx, "evaluate") as run:
        ev = evaluate_predictions(ctx.cfg, ctx.repo)
        h = ctx.repo.prediction_history(100000)
        done = h[h["outcome"].notna()]
        out = {"evaluation": ev, "n_predictions": int(len(h)), "n_evaluated": int(len(done))}
        if len(done):
            done = done.copy()
            for c in ("hit_stop", "hit_tp1", "hit_tp2", "prediction_correct"):
                done[c] = done[c].astype(float)
            out["all"] = {"accuracy": float(done["prediction_correct"].mean()),
                          "avg_predicted_return": float(done["expected_return"].mean()),
                          "avg_actual_return_5d": float(done["actual_return_5d"].mean())}
            buys = done[done["decision"] == "BUY"]
            if len(buys):
                out["buy"] = {"n": int(len(buys)), "hit_tp1": float(buys["hit_tp1"].mean()), "hit_tp2": float(buys["hit_tp2"].mean()),
                              "hit_stop": float(buys["hit_stop"].mean()), "avg_actual_return_5d": float(buys["actual_return_5d"].mean()),
                              "avg_mfe": float(buys["mfe"].mean()), "avg_mae": float(buys["mae"].mean())}
        run.stats["summary"] = out
        return out


def check_history_window(ctx: AppContext, estimate: bool) -> None:
    """Pastikan histori cukup untuk walk-forward SEBELUM pekerjaan berat.

    estimate=True : sebelum mengunduh — kalender diperkirakan (~242 hari bursa/tahun) dari config.
    estimate=False: sesudah mengunduh — memakai tanggal S&P 500 yang benar-benar ada di database.
    """
    import numpy as np

    from app.backtest.walk_forward import SplitError, assert_feasible
    from app.data.ingestion import initial_start
    sym = get(ctx.cfg, "data.index_id", "COMPOSITE")
    if estimate:
        start = initial_start(ctx.cfg, ctx.calendar)
        first_db = ctx.repo.min_index_date(sym)
        if first_db is not None:
            start = min(start, first_db)
        d = pd.bdate_range(start, ctx.calendar.expected_latest_market_date())
        dates = pd.Series(d[np.arange(len(d)) % 14 != 0])
    else:
        dates = pd.Series(ctx.repo.index_dates(sym))
    try:
        assert_feasible(dates, ctx.cfg)
    except SplitError as e:
        raise PipelineAbort(str(e) + (" (diperiksa sebelum mengunduh data)" if estimate else "")) from None


def job_setup(ctx: AppContext, skip_backtest: bool = False) -> dict:
    """First initialization (§11, §66). Idempoten: aman dijalankan ulang."""
    out = {}
    env = check_environment(ctx.cfg)
    if env:
        raise PipelineAbort("Environment tidak valid: " + "; ".join(env))
    ctx.db.scalar("SELECT 1")
    check_history_window(ctx, estimate=True)
    with PipelineRun(ctx, "setup") as run:
        out["ingest"] = _ingest(ctx, run, full_actions=True)
        check_history_window(ctx, estimate=False)
        run.stats["summary"] = {"ingestion": {k: v for k, v in out["ingest"]["ingestion"].items() if k != "failed"}}
    out["validation"] = job_validate(ctx)[["check", "severity", "count"]].to_dict(orient="records")
    out["features"] = job_features(ctx)
    out["earnings_backfill"] = _ensure_earnings_history(ctx)
    from app.models.retrain import retrain
    with PipelineRun(ctx, "retrain") as run:
        df = build_dataset(ctx.cfg, ctx.repo, None, labels=True)
        tr = retrain(ctx.cfg, df, ctx.models, force=True)
        if tr["decision"] == "REJECTED" and ctx.models.active_version() is None:
            # Model pertama: perlu model ACTIVE agar daily bisa berjalan. Ditandai BASELINE dan
            # dilaporkan health check (MODEL STATUS: WARN) — filter NO-TRADE tetap berlaku.
            ctx.models.promote(tr["version"], "BASELINE (gerbang promosi gagal: " + "; ".join(tr["reasons"]) + ")")
            tr["decision"] = "PROMOTED_AS_BASELINE"
        run.stats["model_version"] = tr["version"]
        run.stats["summary"] = {k: v for k, v in tr.items() if k != "metrics"}
        out["train"] = tr
    if not skip_backtest:
        out["backtest"] = job_backtest(ctx, live=False)
    return out
