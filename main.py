"""CLI — Swing Stock Predictor (pasar saham AS).

Production: GitHub Actions menjalankan `python main.py daily` (terjadwal atau manual). Tidak ada scheduler lokal.

  python main.py health            python main.py setup             python main.py daily
  python main.py update-data       python main.py validate-data     python main.py features
  python main.py train             python main.py evaluate          python main.py backtest
  python main.py scan              python main.py predict AAPL      python main.py models
  python main.py make-sample       python main.py db-schema       python main.py db-maintenance [--dry-run] [--full]

Opsi global: --config DIR  --set key=value  --as-of YYYY-MM-DD (simulasi tanggal; untuk test/backfill)
Exit code: 0 sukses, 1 gagal (GitHub Actions → FAILED), 2 config salah, 3 health kritis gagal.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from app.config import ConfigError, get, load_config, resolve_path  # noqa: E402
from app.utils.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("cli")


def _ctx(cfg):
    from app.pipeline.context import AppContext
    return AppContext.create(cfg)


def _print_summary(s: dict) -> None:
    if s.get("data_warning"):
        print("!! " + s["data_warning"])
    f = s["data_freshness"]
    print(f"\n{s['date']} | Regime {s['market_regime']} | Data {f['status']} (DB {f['latest_database_date']}, "
          f"ekspektasi {f['latest_expected_market_date']}) | Model {s['model_version']}")
    print(f"Discan {s['universe_scanned']}, tradeable {s['tradeable']} | {s['decision_counts']}")
    if s.get("coverage"):
        c = s["coverage"]
        print(f"Cakupan: {c['processed']}/{c['total_universe']} dianalisis ({c['coverage_pct']}%), "
              f"layak sinyal {c['signal_eligible']} | {c['status_counts']}")
    print(f"\n{s['headline']}")
    if s["recommendations"]:
        print(f"{'#':>2} {'Ticker':<6} {'Setup':<22} {'Skor':>4} {'P(bull)':>7} {'Entry':>17} {'SL':>9} {'TP1':>9} {'TP2':>9} {'R/R':>5} {'Qty':>5} Decision")
        for r in s["recommendations"]:
            print(f"{r['rank']:>2} {r['ticker']:<6} {r['setup']:<22} {r['score']:>4.0f} {r['probability_bullish']:>7.0%} "
                  f"{r['entry_low']:>8,.2f}-{r['entry_high']:<8,.2f} {r['stop_loss']:>9,.2f} {r['take_profit_1']:>9,.2f} "
                  f"{r['take_profit_2']:>9,.2f} {r['risk_reward']:>5.2f} {r['lots']:>5} {r['decision']}")
    if s.get("watchlist"):
        print("Watchlist: " + ", ".join(f"{w['ticker']}({w['setup']})" for w in s["watchlist"][:8]))
    print(s["disclaimer"])


def cmd_health(cfg, a):
    from app.pipeline.health import format_health, health_check, health_json
    ctx = _ctx(cfg)
    h = health_check(ctx, check_provider=not a.no_provider)
    if getattr(a, "strict", False) and h["checks"].get("DATABASE", {}).get("status") == "OK":
        from app.pipeline.health import write_probe
        st, detail = write_probe(ctx.db)
        h["checks"]["DATABASE WRITE"] = {"status": st, "detail": detail}
    print(health_json(h) if a.json else format_health(h))
    from app.utils.gha import annotate
    annotate("Health", "; ".join(f"{k}: {v['status']} ({str(v.get('detail', ''))[:90]})"
                                 for k, v in h["checks"].items()), "notice" if h["ok"] else "warning")
    if getattr(a, "strict", False):
        from app.pipeline.health import strict_failures
        bad = strict_failures(h)
        if bad:
            print("STRICT HEALTH FAILED: " + ", ".join(bad))
            return 3
    return 0 if h["ok"] else 3


def cmd_setup(cfg, a):
    from app.pipeline.jobs import job_setup
    out = job_setup(_ctx(cfg), skip_backtest=a.skip_backtest)
    tr = out["train"]
    print(f"Setup selesai. Model {tr['version']}: {tr['decision']} {('— ' + '; '.join(tr['reasons'])) if tr.get('reasons') else ''}")
    from app.utils.gha import annotate
    annotate("Setup: model", f"{tr['version']} {tr['decision']} " + "; ".join(tr.get("reasons") or []))
    ing = out["ingest"]["ingestion"]
    print(f"Data: +{ing['rows_inserted']} baris, {ing['stocks_processed']} emiten, gagal {ing['stocks_failed']}")
    if "backtest" in out:
        for k, v in out["backtest"].items():
            if v and "strategy" in v:
                print(f"Backtest {k}: return {v['strategy'].get('total_return', 0):+.1%} vs S&P 500 {v['ihsg_buy_hold'].get('total_return', 0):+.1%}, "
                      f"trades {v['strategy'].get('number_of_trades', 0)}")
    return 0


def cmd_daily(cfg, a):
    from app.pipeline.jobs import job_daily
    s = job_daily(_ctx(cfg), skip_ingest=a.skip_ingest)
    _print_summary(s)
    print(f"\nRun {s['run_id']}: {s['run_status']}")
    from app.utils.gha import annotate
    rec = ", ".join(f"{r['ticker']}({r['decision']})" for r in (s.get("recommendations") or [])[:8]) or "-"
    cov = s.get("coverage") or {}
    annotate("Daily", f"{s['date']} {s['run_status']} | regime {s['market_regime']} | model {s['model_version']} | "
                      f"{s['decision_counts']} | cakupan {cov.get('processed')}/{cov.get('total_universe')} | {s['headline']} | "
                      f"rekomendasi: {rec}")
    return 0


def cmd_update_data(cfg, a):
    from app.pipeline.jobs import job_update_data
    out = job_update_data(_ctx(cfg), full_actions=a.actions)
    i = out["ingestion"]
    print(f"Ekspektasi tanggal bursa: {i['expected_date']} | S&P 500 +{i['index_rows']} | emiten diminta {i['stocks_requested']}, "
          f"up-to-date {i['stocks_up_to_date']}, diproses {i['stocks_processed']}, gagal {i['stocks_failed']}")
    print(f"Baris: +{i['rows_inserted']} baru, {i['rows_updated']} diperbarui, {i['rows_rejected']} ditolak validasi"
          + (f" | unduh ulang (split): {i['refetched']}" if i["refetched"] else ""))
    return 0


def cmd_validate(cfg, a):
    from app.pipeline.jobs import job_validate
    rep = job_validate(_ctx(cfg))
    for r in rep.itertuples():
        print(f"[{r.severity:<5}] {r.check:<36} {r.count:>7}  {r.detail or ''} {list(r.tickers)[:5] if len(r.tickers) else ''}")
    return 0


def cmd_features(cfg, a):
    from app.pipeline.jobs import job_features
    print(json.dumps(job_features(_ctx(cfg)), indent=2))
    return 0


def cmd_train(cfg, a):
    from app.pipeline.jobs import job_train
    out = job_train(_ctx(cfg), force=not a.if_due, compare=not a.skip_compare)
    if out["decision"] == "SKIPPED":
        print(out["reason"])
        from app.utils.gha import annotate
        annotate("Retrain", out["reason"])
        return 0
    m = out["metrics"]
    print(f"{out['version']}: {out['decision']} (sebelumnya aktif: {out['previous_active']})")
    from app.utils.gha import annotate
    annotate("Retrain", f"{out['version']} {out['decision']} | {m['classifiers']} | AUC {m['auc_mean']:.3f}±{m['auc_std']:.3f} "
                        f"| rank IC {m.get('rank_ic_mean') or float('nan'):.4f} | fitur {m.get('n_features')} | "
                        + ("; ".join(out["reasons"]) or "semua gerbang lolos"))
    print(f"Model {m['classifiers']} | AUC {m['auc_mean']:.3f} ± {m['auc_std']:.3f} | ECE {m['ece_bullish']:.3f} | "
          f"fold proxy positif {m['positive_fold_ratio']:.0%}")
    if m.get("head_to_head"):
        print(f"Head-to-head holdout: {m['head_to_head']}")
    for r in out["reasons"]:
        print("  - " + r)
    return 0


def cmd_backtest(cfg, a):
    from app.pipeline.jobs import job_backtest
    out = job_backtest(_ctx(cfg), last_n_folds=a.folds, live=not a.no_live)
    for k, v in out.items():
        if k == "factor_ic_top":
            continue
        if not v:
            print(f"{k}: tidak ada data")
            continue
        s, b = v["strategy"], v.get("ihsg_buy_hold", {})
        t = v.get("technical_only")
        print(f"\n[{k}] {s.get('start')}..{s.get('end')} (setelah biaya)")
        print(f"  Strategi : return {s.get('total_return', 0):+.1%} | CAGR {s.get('cagr', 0):+.1%} | MaxDD {s.get('max_drawdown', 0):.1%} | "
              f"Sharpe {s.get('sharpe', 0):.2f} | trades {s.get('number_of_trades', 0)} | win {s.get('win_rate', float('nan')):.0%} | "
              f"PF {s.get('profit_factor', float('nan')):.2f}")
        if t:
            print(f"  Teknikal : return {t.get('total_return', 0):+.1%} | Sharpe {t.get('sharpe', 0):.2f} | trades {t.get('number_of_trades', 0)}")
        if b:
            print(f"  S&P 500 B&H : return {b.get('total_return', 0):+.1%} | MaxDD {b.get('max_drawdown', 0):.1%} | Sharpe {b.get('sharpe', 0):.2f}")
        sig = (s.get("significance") or {})
        if sig.get("verdict"):
            print(f"  Bukti statistik: {sig['verdict']}")
            if sig.get("beta") is not None:
                print(f"  Pembanding adil (beta {sig['beta']} × S&P 500): {sig.get('beta_matched_index_return', 0):+.1%}")
    from app.utils.gha import annotate
    for k, v in out.items():
        if k == "factor_ic_top" or not v:
            continue
        s, sig = v["strategy"], (v["strategy"].get("significance") or {})
        annotate(f"Backtest {k}", f"{s.get('start')}..{s.get('end')} return {s.get('total_return', 0):+.1%}, Sharpe "
                                  f"{s.get('sharpe', 0):.2f}, trades {s.get('number_of_trades', 0)}, MaxDD "
                                  f"{s.get('max_drawdown', 0):.1%} | beta-matched {sig.get('beta_matched_index_return', 0) or 0:+.1%} "
                                  f"| {sig.get('verdict', 'n/a')}")
    if out.get("factor_ic_top"):
        annotate("Factor IC (top 12, |t|)", " | ".join(
            f"{r['feature']} IC {r['ic_mean']:+.4f} t {r['ic_tstat']:+.2f} 12m {r.get('ic_mean_12m') or float('nan'):+.4f}"
            for r in out["factor_ic_top"][:12]))
    if out.get("factor_ic_top"):
        print("\nRank IC faktor vs return residual 5H (|t| ≥ 2 ≈ signifikan):")
        for r in out["factor_ic_top"][:15]:
            print(f"  {r['feature']:<28} IC {r['ic_mean']:+.4f}  t {r['ic_tstat']:+.2f}  12 bln {r.get('ic_mean_12m') or float('nan'):+.4f}")
    return 0


def cmd_scan(cfg, a):
    from app.pipeline.jobs import job_scan
    s = job_scan(_ctx(cfg), date=a.date, save=not a.no_save)
    _print_summary(s)
    if a.json:
        print(json.dumps(s, indent=2, default=str, ensure_ascii=False))
    return 0


def cmd_predict(cfg, a):
    from app.pipeline.jobs import job_predict
    from app.scanner.scanner import format_analysis
    rec = job_predict(_ctx(cfg), a.ticker, a.date)
    print(format_analysis(rec))
    print(f"Status data: {rec['data_freshness']}")
    if a.json:
        print(json.dumps(rec, indent=2, default=str, ensure_ascii=False))
    return 0


def cmd_evaluate(cfg, a):
    from app.pipeline.jobs import job_evaluate
    print(json.dumps(job_evaluate(_ctx(cfg)), indent=2, default=str))
    return 0


def cmd_models(cfg, a):
    ctx = _ctx(cfg)
    if a.promote:
        ctx.models.promote(a.promote, "manual override")
        print(f"{a.promote} sekarang ACTIVE")
    v = ctx.models.list_versions()
    print(v[["version", "status", "created_at", "train_end", "classifiers", "storage_backend", "artifact_bytes", "notes"]].to_string(index=False)
          if len(v) else "Belum ada model.")
    return 0


def cmd_db_maintenance(cfg, a):
    from app.pipeline.jobs import job_db_maintenance
    out = job_db_maintenance(_ctx(cfg), dry_run=a.dry_run, full=a.full)
    print(f"Ukuran database: {out['before_mb']} MB → {out['after_mb']} MB (batas {out['limit_mb']} MB)"
          + ("   [DRY-RUN: tidak ada yang dihapus]" if out["dry_run"] else ""))
    for s in out["steps"]:
        print(f"  {s['rows']:>9,}  {s['step']}")
    if out["vacuumed"]:
        print("VACUUM: " + ", ".join(out["vacuumed"]))
    if not out["full"] and not out["dry_run"]:
        print("Catatan: ruang terhapus kini dipakai ulang (database berhenti membesar). Angka ukuran baru turun "
              "setelah --full (VACUUM FULL, mengunci tabel sebentar).")
    print("Tabel terbesar: " + ", ".join(f"{t} {mb} MB" for t, mb in list(out["table_mb"].items())[:5]))
    return 0


def cmd_make_sample(cfg, a):
    from app.data.sample import generate_sample_dataset
    out = resolve_path(cfg, a.out)
    generate_sample_dataset(out, n_tickers=a.tickers, start=a.start, end=a.end, seed=a.seed)
    print(f"Dataset CONTOH (sintetis) dibuat di {out}")
    return 0


def cmd_db_schema(cfg, a):
    from app.database.schema import postgres_sql_file
    p = resolve_path(cfg, "sql/schema_postgres.sql")
    p.write_text(postgres_sql_file(), encoding="utf-8")
    print(f"Skema PostgreSQL ditulis ke {p}")
    return 0


def _as_of(cfg, a):
    return getattr(a, "date", None) or cfg.get("_as_of")


def cmd_research(cfg, a):
    from app.data.tickers import normalize_ticker
    from app.research.report import build_research, stock_report
    ctx = _ctx(cfg)
    t = normalize_ticker(a.ticker)
    res = build_research(cfg, ctx.repo, _as_of(cfg, a), tickers=[t])
    print(stock_report(res, t))
    return 0


def cmd_rankings(cfg, a):
    from app.research.integrated_scoring import top
    from app.research.report import build_research
    res = build_research(cfg, _ctx(cfg).repo, _as_of(cfg, a))
    cols = ["ticker", f"score_{a.ranking}", "value_decision", "swing_decision", "valuation_status", "value_trap_risk",
            "foreign_flow_status", "accumulation_status"]
    t = top(res["table"], a.ranking, a.top)
    print(f"Ranking {a.ranking} — {res['as_of']:%Y-%m-%d} ({len(t)} emiten berdata cukup)")
    print(t.reindex(columns=cols).to_string(index=False) if len(t) else "Tidak ada emiten dengan data cukup.")
    return 0


# ------------------------------------------------------------------ model challenger (app/challengers)
def _challenger_models(cfg, arg):
    from app.challengers import MODEL_NAMES, enabled_models
    if not arg or arg == "all":
        return enabled_models(cfg)
    names = [x.strip().lower() for x in arg.split(",") if x.strip()]
    bad = [n for n in names if n not in MODEL_NAMES]
    if bad:
        raise SystemExit(f"Model tidak dikenal: {bad}. Pilihan: {', '.join(MODEL_NAMES)}")
    return names


def cmd_challenger_prepare(cfg, a):
    from app.challengers.runner import prepare_dataset, save_dataset_npz
    ctx = _ctx(cfg)
    df = prepare_dataset(cfg, ctx.repo)
    save_dataset_npz(df, a.out)
    print(f"Dataset challenger: {len(df)} baris, {df['ticker'].nunique()} emiten, {df['date'].min().date()}..{df['date'].max().date()} → {a.out}")
    return 0


def cmd_challenger_train(cfg, a):
    from app.challengers import MODEL_LABELS
    from app.challengers.runner import load_dataset_npz, prepare_dataset, run_train
    from app.pipeline.context import PipelineRun
    ctx = _ctx(cfg)
    df = load_dataset_npz(a.dataset) if a.dataset else prepare_dataset(cfg, ctx.repo)
    rc = 0
    for model in _challenger_models(cfg, a.model):
        with PipelineRun(ctx, f"challenger_{model}") as run:
            out = run_train(model, df, cfg, ctx.repo, run.run_id)
            run.stats["model_version"] = out["version"]
            run.stats["summary"] = {k: v for k, v in out.items() if k != "model"}
        t = out["test"]
        print(f"\n{MODEL_LABELS[model]} {out['version']} — test {t.get('eval_start')}..{t.get('eval_end')} (n={t.get('n')})")
        print(f"  {'ranking_rule':<24} {t.get('ranking_rule', 'prob_bullish')} (dipilih di fold validasi)")
        for k in ("roc_auc_bullish", "auc_up", "log_loss", "log_loss_raw", "ece_bullish", "ece_bullish_raw",
                  "topk_mean_net_return", "topk_by_rule_net", "topk_by_exp_return_net", "ic_mean", "ic_t",
                  "coverage_80", "vol_rank_corr"):
            v = t.get(k)
            print(f"  {k:<24} {'n/a' if v is None else (f'{v:+.2%}' if ('return' in k or 'net' in k) else f'{v:.4f}')}")
        for line in out["verdict"]:
            print(f"  - {line}")
    return rc


def cmd_challenger_predict(cfg, a):
    from app.challengers import challenger_cfg
    from app.challengers.runner import prepare_dataset, run_predict
    from app.pipeline.context import PipelineRun
    ctx = _ctx(cfg)
    models = _challenger_models(cfg, a.models)
    with PipelineRun(ctx, "challenger_predict") as run:
        df = prepare_dataset(cfg, ctx.repo, challenger_cfg(cfg)["predict_lookback_days"], labels=False)
        out = run_predict(models, df, cfg, ctx.repo, run.run_id)
        run.stats["summary"] = out
        if any(v["status"] == "FAILED" for v in out.values()):
            run.status = "PARTIAL_SUCCESS"
    for m, v in out.items():
        print(f"{m:<12} {v['status']:<9} {v.get('version', '')} {v.get('n', '')} {v.get('date', '')} {v.get('detail', '')}")
    return 1 if out and all(v["status"] == "FAILED" for v in out.values()) else 0


def cmd_challenger_report(cfg, a):
    from app.challengers.report import comparison_table
    t = comparison_table(_ctx(cfg).repo)
    print(t.to_string(index=False) if len(t) else "Belum ada model challenger terlatih.")
    return 0


def cmd_sectors_update(cfg, a):
    from app.data.universe import UniverseManager
    ctx = _ctx(cfg)
    out = UniverseManager(cfg, ctx.repo, ctx.chain)._enrich_metadata(limit=a.limit)
    print(f"Sektor: {out['attempted']} emiten dicek, {out['filled']} terisi, {out['missing']} masih tanpa sektor")
    return 0


def cmd_fundamentals_update(cfg, a):
    from app.valuation.fundamental_provider import update_fundamentals
    ctx = _ctx(cfg)
    tickers = [x.strip().upper() for x in a.tickers.split(",")] if a.tickers else None
    st = update_fundamentals(cfg, ctx.repo, _as_of(cfg, a) or pd.Timestamp.today(), tickers,
                             earnings_backfill=bool(getattr(a, "earnings_backfill", False)))
    for n, p in st["providers"].items():
        print(f"  {n:<22} {p['status']:<15} {p['detail']}")
    print(f"Dicek {st['checked']} emiten, {st['new_versions']} versi laporan baru, {st['failed']} tanpa data")
    for k, label in (("earnings_watch", "Pantauan rilis laba"), ("earnings_backfill", "Backfill riwayat laba")):
        if k in st:
            print(f"{label}: {st[k]['checked']} saham dicek, {st[k]['rows']} event 8-K 2.02")
    return 0


def cmd_av_update(cfg, a):
    """Alpha Vantage: kalender rilis, yield Treasury, berita, surprise EPS — dalam batas kuota 24 jam."""
    from app.data.alpha_vantage_update import update_alpha_vantage
    from app.utils.gha import annotate
    ctx = _ctx(cfg)
    try:
        out = update_alpha_vantage(cfg, ctx.repo, _as_of(cfg, a) or pd.Timestamp.today(), mode=a.mode,
                                   max_calls=a.max_calls)
    except Exception as e:  # data tambahan tidak boleh menggagalkan pipeline utama
        msg = f"{type(e).__name__}: {e}"[:300]
        print("Alpha Vantage gagal: " + msg)
        annotate("Alpha Vantage", "GAGAL: " + msg, "warning")
        return 0
    if not out["configured"]:
        print("Alpha Vantage dilewati: GitHub Secret ALPHAVANTAGE_API_KEY belum diset")
        annotate("Alpha Vantage", "dilewati — secret ALPHAVANTAGE_API_KEY belum diset", "warning")
        return 0
    line = (f"{out['calls']} request ({out['mode']}) | kalender +{out['calendar_rows']} baris"
            f" (universe {out.get('calendar_universe', '-')}) | makro +{out['macro_rows']} | berita +{out['news_rows']} | "
            f"surprise EPS {out['earnings_tickers']} saham (+{out['earnings_rows']} kuartal), cakupan "
            f"{out.get('surprise_coverage')} | mutu: {out.get('earnings_with_estimate', 0)}/{out['earnings_rows']} kuartal ber-estimasi, "
            f"{out.get('earnings_report_time', 0)} ber-jam rilis | kuota sisa {out.get('budget_after')}"
            + (f" | berhenti: {out['stopped']}" if out.get("stopped") else ""))
    print(line)
    for e in out["errors"][:10]:
        print("  ! " + e)
    annotate("Alpha Vantage", line + (" | error: " + " / ".join(out["errors"][:5]) if out["errors"] else ""),
             "warning" if out["errors"] and not out["calls"] else "notice")
    return 0


def cmd_fundamentals_template(cfg, a):
    from app.valuation.fundamental_provider import csv_template
    out = resolve_path(cfg, a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Template fundamental: {csv_template(out)} (isi dari laporan keuangan resmi; satuan penuh mata uang laporan)")
    return 0


def cmd_foreign_flow_update(cfg, a):
    from app.flows.foreign_flow_provider import update_foreign_flow
    st = update_foreign_flow(cfg, _ctx(cfg).repo, _as_of(cfg, a) or pd.Timestamp.today(), full=a.full)
    for n, p in st["providers"].items():
        print(f"  {n:<24} {p['status']:<15} {p.get('rows', '')} {p['detail']}")
    print(f"Short flow (FINRA): {st['status']} ({st['rows']} baris)")
    return 0


def cmd_valuation(cfg, a):
    from app.valuation.valuation_report import run_valuation, save_valuations
    ctx = _ctx(cfg)
    d = _as_of(cfg, a) or ctx.repo.max_price_date()
    from app.data.alpha_vantage_update import apply_market_rates
    src = apply_market_rates(cfg, ctx.repo, d)
    if src:
        print(f"Risk-free {cfg['valuation']['risk_free_rate']:.2%} dari {src}")
    df = run_valuation(cfg, ctx.repo, d, use_estimated=a.estimated_pit)
    if not a.no_save and not a.estimated_pit:
        save_valuations(ctx.repo, df)
    print(df["valuation_status"].value_counts().to_string() if len(df) else "Tidak ada emiten aktif")
    return 0


def cmd_factor_report(cfg, a):
    from app.research.factor_report import run_factor_report
    t = run_factor_report(_ctx(cfg))
    if t.empty:
        print("Tidak ada data untuk laporan faktor")
        return 0
    print(f"Rank IC faktor vs {t.attrs.get('target')} (tanggal tidak tumpang tindih; |t| ≥ 2 ≈ signifikan)")
    print(t.head(int(a.top)).round(4).to_string(index=False))
    return 0


def cmd_evaluate_modules(cfg, a):
    from app.research.evaluation import evaluate_modules, format_evaluation
    res = evaluate_modules(cfg, _ctx(cfg).repo, use_estimated=a.estimated_pit, save=not a.no_save)
    print(format_evaluation(res))
    if a.json:
        print(json.dumps(res, default=str, indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Swing Stock Predictor — US (NYSE/Nasdaq)")
    p.add_argument("--config", default=None, help="folder config (default: config/)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--as-of", default=None, help="simulasikan tanggal (YYYY-MM-DD)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("health"); s.add_argument("--json", action="store_true"); s.add_argument("--no-provider", action="store_true")
    s.add_argument("--strict", action="store_true", help="exit 3 juga bila DATABASE SIZE / DATABASE PROVIDER = FAIL (workflow Database Health Check)")
    s.set_defaults(fn=cmd_health)
    s = sub.add_parser("setup"); s.add_argument("--skip-backtest", action="store_true"); s.set_defaults(fn=cmd_setup)
    s = sub.add_parser("daily"); s.add_argument("--skip-ingest", action="store_true"); s.set_defaults(fn=cmd_daily)
    s = sub.add_parser("update-data"); s.add_argument("--actions", action="store_true", help="cek corporate action semua emiten"); s.set_defaults(fn=cmd_update_data)
    sub.add_parser("validate-data").set_defaults(fn=cmd_validate)
    sub.add_parser("features").set_defaults(fn=cmd_features)
    s = sub.add_parser("train"); s.add_argument("--if-due", action="store_true", help="hanya bila model aktif sudah tua")
    s.add_argument("--skip-compare", action="store_true"); s.set_defaults(fn=cmd_train)
    s = sub.add_parser("backtest"); s.add_argument("--folds", type=int, default=None); s.add_argument("--no-live", action="store_true"); s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("scan"); s.add_argument("--date"); s.add_argument("--json", action="store_true"); s.add_argument("--no-save", action="store_true"); s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("predict"); s.add_argument("ticker"); s.add_argument("--date"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_predict)
    sub.add_parser("evaluate").set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("models"); s.add_argument("--promote", metavar="VERSION"); s.set_defaults(fn=cmd_models)
    s = sub.add_parser("make-sample"); s.add_argument("--out", default="data/raw/sample"); s.add_argument("--tickers", type=int, default=45)
    s.add_argument("--start", default="2017-01-02"); s.add_argument("--end", default="2026-10-02"); s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_make_sample)
    sub.add_parser("db-schema").set_defaults(fn=cmd_db_schema)
    s = sub.add_parser("db-maintenance", help="retensi data + VACUUM (cegah kuota storage database terlampaui)")
    s.add_argument("--dry-run", action="store_true", help="tampilkan yang akan dihapus tanpa menghapus")
    s.add_argument("--full", action="store_true", help="VACUUM FULL: ukuran database benar-benar turun")
    s.set_defaults(fn=cmd_db_maintenance)
    s = sub.add_parser("research", help="STOCK RESEARCH REPORT satu emiten"); s.add_argument("ticker"); s.add_argument("--date")
    s.set_defaults(fn=cmd_research)
    s = sub.add_parser("rankings", help="ranking riset terintegrasi"); s.add_argument("--date"); s.add_argument("--top", type=int, default=20)
    s.add_argument("--ranking", default="integrated", choices=["best_value", "quality_value", "accumulation", "foreign_buying",
                   "swing_setup", "value_accumulation", "value_swing", "momentum_flow", "integrated"]); s.set_defaults(fn=cmd_rankings)
    s = sub.add_parser("challenger-prepare", help="bangun dataset challenger sekali (npz, tanpa pickle)")
    s.add_argument("--out", default="challenger_dataset.npz"); s.set_defaults(fn=cmd_challenger_prepare)
    s = sub.add_parser("challenger-train", help="latih & evaluasi model challenger (lstm, gru, tft, arima_garch)")
    s.add_argument("--model", default="all"); s.add_argument("--dataset", help="npz dari challenger-prepare")
    s.set_defaults(fn=cmd_challenger_train)
    s = sub.add_parser("challenger-predict", help="prakiraan harian semua model challenger READY")
    s.add_argument("--models", default="all"); s.set_defaults(fn=cmd_challenger_predict)
    s = sub.add_parser("challenger-report", help="perbandingan challenger vs model aktif vs baseline")
    s.set_defaults(fn=cmd_challenger_report)
    s = sub.add_parser("sectors-update", help="isi sektor saham yang kosong (Yahoo), antrean berputar")
    s.add_argument("--limit", type=int, default=1000); s.set_defaults(fn=cmd_sectors_update)
    s = sub.add_parser("fundamentals-update"); s.add_argument("--tickers", help="AAPL,MSFT (default: incremental)"); s.add_argument("--date")
    s.add_argument("--earnings-backfill", action="store_true", help="ambil riwayat 8-K laba untuk saham yang belum punya")
    s.set_defaults(fn=cmd_fundamentals_update)
    s = sub.add_parser("av-update", help="Alpha Vantage: kalender rilis, yield Treasury, surprise EPS (kuota 25/hari)")
    s.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    s.add_argument("--max-calls", type=int, default=None, help="batasi jumlah request run ini")
    s.add_argument("--date")
    s.set_defaults(fn=cmd_av_update)
    s = sub.add_parser("fundamentals-template"); s.add_argument("--out", default="data/raw/fundamentals/template.csv")
    s.set_defaults(fn=cmd_fundamentals_template)
    s = sub.add_parser("foreign-flow-update"); s.add_argument("--full", action="store_true"); s.add_argument("--date")
    s.set_defaults(fn=cmd_foreign_flow_update)
    s = sub.add_parser("valuation"); s.add_argument("--date"); s.add_argument("--no-save", action="store_true")
    s.add_argument("--estimated-pit", action="store_true", help="riset: estimasi tanggal publikasi (berlabel ESTIMATED_PIT, tidak disimpan)")
    s.set_defaults(fn=cmd_valuation)
    s = sub.add_parser("factor-report", help="rank IC setiap fitur vs return residual 5H (riset faktor)")
    s.add_argument("--top", type=int, default=40)
    s.set_defaults(fn=cmd_factor_report)
    s = sub.add_parser("evaluate-modules", help="evaluasi historis 9 varian + IC faktor")
    s.add_argument("--estimated-pit", action="store_true"); s.add_argument("--no-save", action="store_true"); s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_evaluate_modules)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    if args.as_of:
        cfg["_as_of"] = args.as_of
    setup_logging(get(cfg, "project.log_level", "INFO"), resolve_path(cfg, "data/cache/logs"))
    try:
        return int(args.fn(cfg, args) or 0)
    except Exception as e:  # error handling terpusat; traceback lengkap ada di log & system_logs
        log.exception("Perintah '%s' gagal: %s", args.cmd, e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
