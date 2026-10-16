"""Pemeliharaan database: retensi data + VACUUM (§55 cost control).

Kenapa perlu: semua paket PostgreSQL gratis punya batas storage kecil, dan melewatinya MEMBLOKIR penulisan
(Neon Free 1 GB/project: write diblokir; Aiven Free 1 GB disk; Supabase Free 500 MB: read-only). Daily pipeline
lalu gagal setiap hari. Batas efektif per provider ada di app/database/providers.py (override: DB_SIZE_LIMIT_MB).

Diukur di PostgreSQL 16 untuk ~920 emiten:
  predictions + evaluasi ≈ 800 B/baris → ±177 MB/tahun   (pertumbuhan TERBESAR: semua emiten, setiap hari)
  price_history          ≈ 200 B/baris → ±45 MB/tahun
Kebijakan retensi (config/app.yaml → retention):
  - prediksi WAIT/AVOID lebih tua dari N hari dihapus (beserta evaluasinya); BUY & WATCHLIST disimpan
    selamanya → rekam jejak sinyal tidak pernah hilang;
  - report harian di database > N hari dihapus (salinan juga ada sebagai artifact GitHub Actions);
  - system_logs > N hari dihapus;
  - price_history lebih tua dari N tahun dihapus (N >= data.initial_history_years, dan cukup untuk walk-forward);
  - file model versi lama (bukan ACTIVE, di luar N terbaru) dihapus — registry & metriknya tetap;
  - valuation_results harian > valuation.store_daily_days dihapus KECUALI valuasi akhir bulan (histori evaluasi);
  - accumulation_signals > N hari, foreign_flow_history > N tahun (research_retention);
  - financial_statements TIDAK pernah dihapus (histori point-in-time & restatement).
Catatan PostgreSQL: DELETE + VACUUM membuat ruang bisa dipakai ulang (database berhenti membesar), tetapi
angka ukuran database baru TURUN setelah VACUUM FULL (mengunci tabel, butuh ruang disk sementara).
"""
from __future__ import annotations

import pandas as pd

from app.config import get
from app.database.db import Database
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

SIGNAL_DECISIONS = ("BUY", "WATCHLIST")


# --------------------------------------------------------------------------------------------- ukuran
def database_size_bytes(db: Database) -> int:
    if db.dialect == "postgres":
        try:  # jumlah semua database di cluster (konservatif: Neon/Supabase menghitung kuota per project)
            return int(db.scalar("SELECT SUM(pg_database_size(datname)) FROM pg_database"))
        except Exception:
            db.conn.rollback()
            return int(db.scalar("SELECT pg_database_size(current_database())"))
    pages = db.scalar("PRAGMA page_count") or 0
    return int(pages) * int(db.scalar("PRAGMA page_size") or 4096)


def table_sizes(db: Database) -> dict[str, int]:
    from app.database.schema import TABLE_NAMES
    if db.dialect == "postgres":
        rows = db.query("SELECT c.relname, pg_total_relation_size(c.oid) FROM pg_class c JOIN pg_namespace n "
                        "ON n.oid = c.relnamespace WHERE n.nspname = current_schema() AND c.relkind = 'r'")
        return {r[0]: int(r[1]) for r in rows if r[0] in TABLE_NAMES}
    try:
        rows = db.query("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name")
        return {r[0]: int(r[1]) for r in rows if r[0] in TABLE_NAMES}
    except Exception:  # SQLite tanpa dbstat → tidak ada rincian per tabel
        return {}


def is_read_only(db: Database) -> bool:
    if db.dialect != "postgres":
        return False
    try:
        return str(db.scalar("SHOW default_transaction_read_only")).lower() == "on"
    except Exception:
        db.conn.rollback()
        return False


def size_status(cfg: dict, db: Database) -> dict:
    from app.database.providers import resolve_size_limit_mb
    limit = resolve_size_limit_mb(get(cfg, "database.size_limit_mb", "auto"), getattr(db, "url", None)) * 1e6
    size = database_size_bytes(db)
    ratio = size / limit if limit else 0.0
    warn, auto = float(get(cfg, "database.size_warn_ratio", 0.8)), float(get(cfg, "database.auto_maintenance_ratio", 0.85))
    status = "FAIL" if ratio >= 0.95 else "WARN" if ratio >= warn else "OK"
    return {"bytes": size, "limit_bytes": limit, "ratio": ratio, "status": status, "maintenance_due": ratio >= auto,
            "read_only": is_read_only(db), "provider": getattr(db, "provider", db.dialect)}


# --------------------------------------------------------------------------------------------- retensi
def _cutoffs(cfg: dict, today: pd.Timestamp) -> dict:
    r = cfg.get("retention", {})
    return {
        "predictions": (today - pd.Timedelta(days=int(r.get("predictions_nonsignal_days", 180)))).strftime("%Y-%m-%d"),
        "reports": (today - pd.Timedelta(days=int(r.get("reports_days", 60)))).strftime("%Y-%m-%d"),
        "logs": (today - pd.Timedelta(days=int(r.get("system_logs_days", 90)))).strftime("%Y-%m-%d"),
        "prices": (today - pd.DateOffset(years=int(r.get("price_history_years", 7)))).strftime("%Y-%m-%d"),
        "valuation_daily": (today - pd.Timedelta(days=int(get(cfg, "valuation.store_daily_days", 14)))).strftime("%Y-%m-%d"),
        "accumulation": (today - pd.Timedelta(days=int(get(cfg, "research_retention.accumulation_signals_days", 365))))
        .strftime("%Y-%m-%d"),
        "foreign_flow": (today - pd.DateOffset(years=int(get(cfg, "research_retention.foreign_flow_years", 7))))
        .strftime("%Y-%m-%d"),
        "coverage": (today - pd.Timedelta(days=int(r.get("scan_coverage_days", 120)))).strftime("%Y-%m-%d"),
        "forecasts": (today - pd.Timedelta(days=int(r.get("model_forecasts_days", 180)))).strftime("%Y-%m-%d"),
    }


def _valuation_daily_dates(db: Database, cutoff: str) -> list[str]:
    """Tanggal valuasi lebih tua dari cutoff yang BUKAN tanggal valuasi terakhir di bulannya (akhir bulan disimpan)."""
    try:
        d = db.query_df("SELECT DISTINCT as_of_date FROM valuation_results")
    except Exception:
        return []
    if d.empty:
        return []
    dates = pd.to_datetime(d["as_of_date"])
    keep = set(dates.groupby(dates.dt.to_period("M")).max())         # akhir bulan dihitung dari SEMUA tanggal
    return sorted(x.strftime("%Y-%m-%d") for x in dates if x not in keep and x < pd.Timestamp(cutoff))


def _plan(cfg: dict, db: Database, today: pd.Timestamp) -> list[dict]:
    c = _cutoffs(cfg, today)
    nonsig = "decision NOT IN ('BUY', 'WATCHLIST')"
    keep_models = int(cfg.get("retention", {}).get("model_artifacts_keep", 3))
    old_models = ("status <> 'ACTIVE' AND artifact IS NOT NULL AND version NOT IN (SELECT version FROM model_versions "
                  f"WHERE status <> 'ACTIVE' ORDER BY created_at DESC, version DESC LIMIT {keep_models})")
    pred_sub = f"SELECT id FROM predictions WHERE prediction_date < ? AND {nonsig}"
    vdates = _valuation_daily_dates(db, c["valuation_daily"])
    val_steps = []
    for i in range(0, len(vdates), 200):
        chunk = vdates[i:i + 200]
        ph = ", ".join(["?"] * len(chunk))
        val_steps.append({"name": f"valuasi harian {chunk[0]}..{chunk[-1]} (akhir bulan disimpan)",
                          "table": "valuation_results",
                          "count": f"SELECT COUNT(*) FROM valuation_results WHERE as_of_date IN ({ph})",
                          "sql": f"DELETE FROM valuation_results WHERE as_of_date IN ({ph})", "params": chunk})
    return [
        {"name": f"evaluasi prediksi WAIT/AVOID sebelum {c['predictions']}", "table": "prediction_evaluations",
         "count": f"SELECT COUNT(*) FROM prediction_evaluations WHERE prediction_id IN ({pred_sub})",
         "sql": f"DELETE FROM prediction_evaluations WHERE prediction_id IN ({pred_sub})", "params": [c["predictions"]]},
        {"name": f"prediksi WAIT/AVOID sebelum {c['predictions']} (BUY/WATCHLIST disimpan)", "table": "predictions",
         "count": f"SELECT COUNT(*) FROM predictions WHERE prediction_date < ? AND {nonsig}",
         "sql": f"DELETE FROM predictions WHERE prediction_date < ? AND {nonsig}", "params": [c["predictions"]]},
        {"name": f"report harian sebelum {c['reports']}", "table": "reports",
         "count": "SELECT COUNT(*) FROM reports WHERE report_date < ?",
         "sql": "DELETE FROM reports WHERE report_date < ?", "params": [c["reports"]]},
        {"name": f"system log sebelum {c['logs']}", "table": "system_logs",
         "count": "SELECT COUNT(*) FROM system_logs WHERE ts < ?",
         "sql": "DELETE FROM system_logs WHERE ts < ?", "params": [c["logs"]]},
        {"name": f"harga sebelum {c['prices']}", "table": "price_history",
         "count": "SELECT COUNT(*) FROM price_history WHERE date < ?",
         "sql": "DELETE FROM price_history WHERE date < ?", "params": [c["prices"]]},
        {"name": f"sinyal akumulasi sebelum {c['accumulation']}", "table": "accumulation_signals",
         "count": "SELECT COUNT(*) FROM accumulation_signals WHERE date < ?",
         "sql": "DELETE FROM accumulation_signals WHERE date < ?", "params": [c["accumulation"]]},
        {"name": f"foreign flow sebelum {c['foreign_flow']}", "table": "foreign_flow_history",
         "count": "SELECT COUNT(*) FROM foreign_flow_history WHERE date < ?",
         "sql": "DELETE FROM foreign_flow_history WHERE date < ?", "params": [c["foreign_flow"]]},
        {"name": f"prakiraan model challenger sebelum {c['forecasts']}", "table": "model_forecasts",
         "count": "SELECT COUNT(*) FROM model_forecasts WHERE forecast_date < ?",
         "sql": "DELETE FROM model_forecasts WHERE forecast_date < ?", "params": [c["forecasts"]]},
        {"name": f"cakupan scan per emiten sebelum {c['coverage']}", "table": "scan_coverage",
         "count": "SELECT COUNT(*) FROM scan_coverage WHERE scan_date < ?",
         "sql": "DELETE FROM scan_coverage WHERE scan_date < ?", "params": [c["coverage"]]},
        *val_steps,
        {"name": f"file model lama (selain ACTIVE & {keep_models} terbaru)", "table": "model_versions",
         "count": f"SELECT COUNT(*) FROM model_versions WHERE {old_models}",
         "sql": f"UPDATE model_versions SET artifact = NULL, notes = COALESCE(notes, '') || ' [file model dihapus retensi]' "
                f"WHERE {old_models}", "params": []},
    ]


def _vacuum(db: Database, tables: list[str], full: bool, headroom_bytes: float | None) -> list[str]:
    """VACUUM tidak boleh di dalam transaksi → autocommit sementara."""
    done = []
    if db.dialect == "sqlite":
        db.conn.commit()
        db.conn.execute("VACUUM")
        return ["(seluruh database)"]
    conn = db.conn
    conn.commit()
    prev = conn.autocommit
    conn.autocommit = True
    try:
        sizes = table_sizes(db)
        for t in sorted(set(tables), key=lambda x: sizes.get(x, 0)):
            if full and headroom_bytes is not None and sizes.get(t, 0) > 0.8 * headroom_bytes:
                log.warning("VACUUM FULL %s dilewati: butuh ruang sementara ±%.0f MB, tersedia ±%.0f MB",
                            t, sizes[t] / 1e6, headroom_bytes / 1e6)
                conn.execute(f"VACUUM (ANALYZE) {t}")
                done.append(t)
                continue
            conn.execute(f"VACUUM (FULL, ANALYZE) {t}" if full else f"VACUUM (ANALYZE) {t}")
            done.append(t + (" (FULL)" if full else ""))
    finally:
        conn.autocommit = prev
    return done


def run_maintenance(cfg: dict, db: Database, today, dry_run: bool = False, full: bool = False) -> dict:
    today = pd.Timestamp(today).normalize()
    before = size_status(cfg, db)
    if db.dialect == "postgres":
        # Mode read-only (Supabase) / transaksi default read-only: izinkan penulisan di sesi ini agar data bisa dihapus
        db.conn.commit()
        db.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
    steps = []
    for p in _plan(cfg, db, today):
        n = int(db.scalar(p["count"], p["params"]) or 0)
        if n and not dry_run:
            db.execute(p["sql"], p["params"])
        steps.append({"step": p["name"], "table": p["table"], "rows": n})
        if n:
            log.info("%s %s: %d baris", "Akan dihapus" if dry_run else "Dihapus", p["name"], n, extra={"persist": True})
    touched = [s["table"] for s in steps if s["rows"]]
    vacuumed = []
    if not dry_run and (touched or full):
        disk = float(get(cfg, "database.disk_limit_mb", 1024)) * 1e6
        vacuumed = _vacuum(db, touched or list(table_sizes(db)), full, disk - before["bytes"])
    after = size_status(cfg, db)
    out = {"dry_run": dry_run, "full": full, "steps": steps, "vacuumed": vacuumed,
           "before_mb": round(before["bytes"] / 1e6, 1), "after_mb": round(after["bytes"] / 1e6, 1),
           "limit_mb": round(before["limit_bytes"] / 1e6), "read_only_before": before["read_only"],
           "read_only_after": after["read_only"], "table_mb": {t: round(s / 1e6, 1) for t, s in
                                                               sorted(table_sizes(db).items(), key=lambda x: -x[1])}}
    log.info("Maintenance database: %.1f MB → %.1f MB (batas %d MB)%s", out["before_mb"], out["after_mb"], out["limit_mb"],
             " [dry-run]" if dry_run else "", extra={"persist": True})
    return out
