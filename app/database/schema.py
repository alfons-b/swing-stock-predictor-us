"""Skema database (§20) — didefinisikan sekali, di-render ke SQLite atau PostgreSQL.

Tipe logis → fisik:
  ID     : INTEGER PRIMARY KEY AUTOINCREMENT | BIGSERIAL PRIMARY KEY
  TEXT, INT(BIGINT), REAL(DOUBLE PRECISION), DATE (TEXT ISO | DATE), TS (TEXT ISO | TIMESTAMPTZ),
  BOOL (INTEGER 0/1 | BOOLEAN), JSON (TEXT | JSONB→disimpan TEXT agar portabel), BLOB (BLOB | BYTEA)
Migrasi bersifat additive & idempoten (CREATE ... IF NOT EXISTS), dicatat di `schema_migrations`.
"""
from __future__ import annotations

from app.database.db import Database

SCHEMA_VERSION = 9   # 8: earnings_events (SEC 8-K Item 2.02) · 9: Alpha Vantage (surprise EPS, kalender, makro, berita, kuota)

TYPES = {
    "sqlite": {"ID": "INTEGER PRIMARY KEY AUTOINCREMENT", "TEXT": "TEXT", "INT": "INTEGER", "REAL": "REAL",
               "DATE": "TEXT", "TS": "TEXT", "BOOL": "INTEGER", "JSON": "TEXT", "BLOB": "BLOB"},
    "postgres": {"ID": "BIGSERIAL PRIMARY KEY", "TEXT": "TEXT", "INT": "BIGINT", "REAL": "DOUBLE PRECISION",
                 "DATE": "DATE", "TS": "TIMESTAMPTZ", "BOOL": "BOOLEAN", "JSON": "TEXT", "BLOB": "BYTEA"},
}

# (nama_tabel, [(kolom, tipe, constraint)], [unique keys], [index columns])
TABLES = [
    ("stocks", [("id", "ID", ""), ("ticker", "TEXT", "NOT NULL"), ("name", "TEXT", ""), ("sector", "TEXT", ""),
                ("subsector", "TEXT", ""), ("board", "TEXT", ""), ("listing_date", "DATE", ""),
                ("delisting_date", "DATE", ""), ("is_active", "BOOL", "NOT NULL"), ("previous_ticker", "TEXT", ""), ("listed_shares", "REAL", ""), ("fundamentals_checked_at", "TS", ""),
                ("sector_checked_at", "TS", ""),
                ("first_seen", "TS", ""), ("last_seen", "TS", ""), ("updated_at", "TS", "")],
     [["ticker"]], [["is_active"], ["sector"]]),
    ("price_history", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"), ("date", "DATE", "NOT NULL"),
                       ("open", "REAL", ""), ("high", "REAL", ""), ("low", "REAL", ""), ("close", "REAL", ""),
                       ("volume", "REAL", ""), ("value", "REAL", ""), ("frequency", "REAL", ""),
                       ("is_adjusted", "BOOL", ""), ("source", "TEXT", ""), ("ingested_at", "TS", "")],
     [["stock_id", "date"]], [["date"]]),
    ("corporate_actions", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"), ("ex_date", "DATE", "NOT NULL"),
                           ("action", "TEXT", "NOT NULL"), ("ratio", "REAL", ""), ("amount", "REAL", ""),
                           ("source", "TEXT", ""), ("processed_at", "TS", "")],
     [["stock_id", "ex_date", "action"]], [["ex_date"]]),
    ("market_index", [("id", "ID", ""), ("symbol", "TEXT", "NOT NULL"), ("date", "DATE", "NOT NULL"), ("open", "REAL", ""),
                      ("high", "REAL", ""), ("low", "REAL", ""), ("close", "REAL", ""), ("volume", "REAL", ""),
                      ("source", "TEXT", ""), ("ingested_at", "TS", "")],
     [["symbol", "date"]], [["date"]]),
    ("sector_data", [("id", "ID", ""), ("date", "DATE", "NOT NULL"), ("sector", "TEXT", "NOT NULL"), ("ret5", "REAL", ""),
                     ("ret20", "REAL", ""), ("ret60", "REAL", ""), ("breadth", "REAL", ""), ("volume_mom", "REAL", ""),
                     ("rs20", "REAL", ""), ("score", "REAL", ""), ("rank", "INT", ""), ("n_stocks", "INT", "")],
     [["date", "sector"]], [["date"]]),
    ("features", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"), ("date", "DATE", "NOT NULL"),
                  ("feature_version", "TEXT", "NOT NULL"), ("payload", "JSON", "")],
     [["stock_id", "date", "feature_version"]], [["date"]]),
    ("predictions", [("id", "ID", ""), ("prediction_date", "DATE", "NOT NULL"), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                     ("ticker", "TEXT", "NOT NULL"), ("model_version", "TEXT", "NOT NULL"), ("horizon", "INT", ""),
                     ("decision", "TEXT", ""), ("setup", "TEXT", ""), ("score", "REAL", ""), ("market_regime", "TEXT", ""),
                     ("prob_bearish", "REAL", ""), ("prob_neutral", "REAL", ""), ("prob_bullish", "REAL", ""),
                     ("expected_return", "REAL", ""), ("prob_hit_tp", "REAL", ""), ("entry_low", "REAL", ""),
                     ("entry_ideal", "REAL", ""), ("entry_high", "REAL", ""), ("stop_loss", "REAL", ""), ("tp1", "REAL", ""),
                     ("tp2", "REAL", ""), ("risk_reward", "REAL", ""), ("position_size", "INT", ""), ("lots", "INT", ""),
                     ("capital_required", "REAL", ""), ("estimated_loss", "REAL", ""), ("confidence", "TEXT", ""),
                     ("close_price", "REAL", ""), ("data_status", "TEXT", ""), ("reasons", "JSON", ""), ("risks", "JSON", ""),
                     ("reject_reasons", "JSON", ""), ("run_id", "TEXT", ""), ("created_at", "TS", ""),
                     ("value_score", "REAL", ""), ("quality_score", "REAL", ""), ("value_trap_risk", "TEXT", ""),
                     ("margin_of_safety", "REAL", ""), ("valuation_status", "TEXT", ""), ("foreign_flow_score", "REAL", ""),
                     ("foreign_flow_status", "TEXT", ""), ("accumulation_score", "REAL", ""), ("distribution_risk", "REAL", ""),
                     ("accumulation_status", "TEXT", ""), ("accumulation_stage", "TEXT", "")],
     [["prediction_date", "stock_id", "model_version"]], [["ticker"], ["decision"], ["model_version"]]),
    ("prediction_evaluations", [("id", "ID", ""), ("prediction_id", "INT", "NOT NULL REFERENCES predictions(id)"),
                                ("evaluated_at", "TS", ""), ("actual_return_3d", "REAL", ""), ("actual_return_5d", "REAL", ""),
                                ("actual_return_10d", "REAL", ""), ("mfe", "REAL", ""), ("mae", "REAL", ""),
                                ("entry_filled", "BOOL", ""), ("hit_stop", "BOOL", ""), ("hit_tp1", "BOOL", ""),
                                ("hit_tp2", "BOOL", ""), ("actual_class", "INT", ""), ("prediction_correct", "BOOL", ""),
                                ("outcome", "TEXT", "")],
     [["prediction_id"]], []),
    ("trading_signals", [("id", "ID", ""), ("signal_date", "DATE", "NOT NULL"), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                         ("ticker", "TEXT", "NOT NULL"), ("rank", "INT", ""), ("decision", "TEXT", ""), ("setup", "TEXT", ""),
                         ("score", "REAL", ""), ("prob_bullish", "REAL", ""), ("expected_return", "REAL", ""),
                         ("entry_low", "REAL", ""), ("entry_ideal", "REAL", ""), ("entry_high", "REAL", ""),
                         ("stop_loss", "REAL", ""), ("tp1", "REAL", ""), ("tp2", "REAL", ""), ("risk_reward", "REAL", ""),
                         ("lots", "INT", ""), ("confidence", "TEXT", ""), ("model_version", "TEXT", ""), ("run_id", "TEXT", "")],
     [["signal_date", "stock_id"]], [["signal_date"]]),
    ("backtest_runs", [("id", "ID", ""), ("run_id", "TEXT", "NOT NULL"), ("created_at", "TS", ""), ("kind", "TEXT", ""),
                       ("period_start", "DATE", ""), ("period_end", "DATE", ""), ("model_version", "TEXT", ""),
                       ("config_hash", "TEXT", ""), ("metrics", "JSON", ""), ("benchmark", "JSON", ""), ("status", "TEXT", "")],
     [["run_id"]], [["created_at"]]),
    ("backtest_trades", [("id", "ID", ""), ("backtest_run_id", "TEXT", "NOT NULL"), ("ticker", "TEXT", ""), ("setup", "TEXT", ""),
                         ("entry_date", "DATE", ""), ("exit_date", "DATE", ""), ("entry_price", "REAL", ""),
                         ("exit_price", "REAL", ""), ("shares", "INT", ""), ("net_pnl", "REAL", ""), ("net_return", "REAL", ""),
                         ("r_multiple", "REAL", ""), ("exit_reason", "TEXT", ""), ("fees", "REAL", "")],
     [["backtest_run_id", "ticker", "entry_date"]], [["backtest_run_id"]]),
    ("model_versions", [("id", "ID", ""), ("version", "TEXT", "NOT NULL"), ("status", "TEXT", "NOT NULL"), ("created_at", "TS", ""),
                        ("promoted_at", "TS", ""), ("train_start", "DATE", ""), ("train_end", "DATE", ""),
                        ("feature_version", "TEXT", ""), ("config_hash", "TEXT", ""), ("classifiers", "TEXT", ""),
                        ("metrics", "JSON", ""), ("storage_backend", "TEXT", ""), ("storage_uri", "TEXT", ""),
                        ("artifact", "BLOB", ""), ("artifact_sha256", "TEXT", ""), ("artifact_bytes", "INT", ""), ("notes", "TEXT", "")],
     [["version"]], [["status"]]),
    ("pipeline_runs", [("id", "ID", ""), ("run_id", "TEXT", "NOT NULL"), ("run_type", "TEXT", "NOT NULL"), ("started_at", "TS", ""),
                       ("finished_at", "TS", ""), ("status", "TEXT", ""), ("stocks_processed", "INT", ""),
                       ("stocks_failed", "INT", ""), ("rows_inserted", "INT", ""), ("rows_updated", "INT", ""),
                       ("model_version", "TEXT", ""), ("data_status", "TEXT", ""), ("market_date", "DATE", ""),
                       ("trigger", "TEXT", ""), ("summary", "JSON", ""), ("error_message", "TEXT", ""), ("created_at", "TS", "")],
     [["run_id"]], [["run_type", "started_at"]]),
    ("data_sources", [("id", "ID", ""), ("name", "TEXT", "NOT NULL"), ("type", "TEXT", ""), ("priority", "INT", ""),
                      ("enabled", "BOOL", ""), ("last_success_at", "TS", ""), ("last_error_at", "TS", ""),
                      ("last_error", "TEXT", ""), ("rows_fetched_total", "INT", ""), ("category", "TEXT", ""),
                      ("status", "TEXT", ""), ("detail", "TEXT", ""), ("checked_at", "TS", "")],
     [["name"]], []),
    ("system_logs", [("id", "ID", ""), ("run_id", "TEXT", ""), ("ts", "TS", ""), ("level", "TEXT", ""), ("logger", "TEXT", ""),
                     ("message", "TEXT", "")],
     [], [["run_id"], ["ts"]]),
    ("reports", [("id", "ID", ""), ("report_date", "DATE", "NOT NULL"), ("kind", "TEXT", "NOT NULL"), ("filename", "TEXT", "NOT NULL"),
                 ("content", "BLOB", ""), ("content_type", "TEXT", ""), ("storage_backend", "TEXT", ""),
                 ("storage_uri", "TEXT", ""), ("run_id", "TEXT", ""), ("created_at", "TS", "")],
     [["report_date", "kind", "filename"]], [["report_date"]]),
    # ---------------- riset: fundamental (point-in-time), valuasi, foreign flow, akumulasi
    # Satu baris = satu laporan (periode × tipe × sumber × versi). `first_known_date` = tanggal paling awal data
    # boleh dipakai (max(tanggal publikasi, tanggal pertama sistem mengetahuinya)); restatement → versi baru.
    ("financial_statements", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                              ("period_end", "DATE", "NOT NULL"), ("period_type", "TEXT", "NOT NULL"), ("fiscal_year", "INT", ""),
                              ("currency", "TEXT", ""), ("items", "JSON", ""), ("items_hash", "TEXT", ""),
                              ("publication_date", "DATE", ""), ("first_known_date", "DATE", "NOT NULL"),
                              ("estimated_available_date", "DATE", ""),
                              ("retrieved_at", "TS", ""), ("source", "TEXT", "NOT NULL"), ("version", "INT", "NOT NULL"),
                              ("quality_status", "TEXT", ""), ("quality_notes", "TEXT", "")],
     [["stock_id", "period_end", "period_type", "source", "version"]], [["stock_id", "first_known_date"]]),
    ("valuation_results", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"), ("as_of_date", "DATE", "NOT NULL"),
                           ("price", "REAL", ""), ("sector_type", "TEXT", ""), ("fair_value_low", "REAL", ""),
                           ("fair_value_base", "REAL", ""), ("fair_value_high", "REAL", ""), ("margin_of_safety", "REAL", ""),
                           ("margin_of_safety_conservative", "REAL", ""), ("valuation_status", "TEXT", ""),
                           ("valuation_confidence", "TEXT", ""), ("value_score", "REAL", ""), ("quality_score", "REAL", ""),
                           ("value_trap_risk", "TEXT", ""), ("methods", "JSON", ""), ("metrics", "JSON", ""),
                           ("value_trap_reasons", "JSON", ""), ("fundamentals_known_date", "DATE", ""),
                           ("fundamentals_period_end", "DATE", ""), ("data_status", "TEXT", ""), ("run_id", "TEXT", ""),
                           ("created_at", "TS", "")],
     [["stock_id", "as_of_date"]], [["as_of_date"]]),
    # Flow mentah per saham per hari (AS: FINRA short volume — sell=short, buy=total−short). Saham dan nilai (USD) DISIMPAN TERPISAH; nilai hanya diisi
    # bila sumber benar-benar memberikannya (value_type ACTUAL). Estimasi lembar × harga TIDAK disimpan di sini.
    ("foreign_flow_history", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"), ("date", "DATE", "NOT NULL"),
                              ("market_segment", "TEXT", "NOT NULL"), ("foreign_buy_shares", "REAL", ""),
                              ("foreign_sell_shares", "REAL", ""), ("net_foreign_shares", "REAL", ""),
                              ("foreign_buy_value", "REAL", ""), ("foreign_sell_value", "REAL", ""),
                              ("net_foreign_value", "REAL", ""), ("value_type", "TEXT", ""), ("total_volume_shares", "REAL", ""),
                              ("units", "TEXT", ""), ("source", "TEXT", ""), ("source_timestamp", "TS", ""),
                              ("retrieved_at", "TS", ""), ("quality_status", "TEXT", ""), ("quality_notes", "TEXT", "")],
     [["stock_id", "date", "market_segment"]], [["date"]]),
    ("accumulation_signals", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"), ("date", "DATE", "NOT NULL"),
                              ("accumulation_score", "REAL", ""), ("distribution_risk", "REAL", ""), ("status", "TEXT", ""),
                              ("stage", "TEXT", ""), ("confidence", "TEXT", ""), ("components", "JSON", ""),
                              ("evidence", "JSON", ""), ("foreign_flow_status", "TEXT", ""), ("run_id", "TEXT", "")],
     [["stock_id", "date"]], [["date"]]),
    # Cakupan pemindaian per emiten per tanggal (app/scanner/coverage.py). ±120 B/baris × ±960 emiten × ±245 hari
    # ≈ 28 MB/tahun → dibatasi retention.scan_coverage_days. Ringkasan per run tetap ada di pipeline_runs.summary.
    ("scan_coverage", [("id", "ID", ""), ("scan_date", "DATE", "NOT NULL"), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                       ("ticker", "TEXT", "NOT NULL"), ("status", "TEXT", "NOT NULL"), ("reason", "TEXT", ""),
                       ("last_data_date", "DATE", ""), ("history_rows", "INT", ""), ("run_id", "TEXT", ""),
                       ("created_at", "TS", "")],
     [["scan_date", "stock_id"]], [["scan_date", "status"]]),
    # Model challenger (app/challengers): registry terpisah dari model_versions agar model aktif tidak terpengaruh.
    ("challenger_models", [("id", "ID", ""), ("model_name", "TEXT", "NOT NULL"), ("version", "TEXT", "NOT NULL"),
                           ("status", "TEXT", "NOT NULL"), ("created_at", "TS", ""), ("train_start", "DATE", ""),
                           ("train_end", "DATE", ""), ("eval_start", "DATE", ""), ("eval_end", "DATE", ""),
                           ("params", "JSON", ""), ("metrics", "JSON", ""), ("artifact", "BLOB", ""),
                           ("run_id", "TEXT", ""), ("notes", "TEXT", "")],
     [["model_name", "version"]], [["model_name", "status"]]),
    # Prakiraan harian challenger: ±960 emiten × 4 model × ±245 hari ≈ 0,94 jt baris/tahun → retention.model_forecasts_days
    ("model_forecasts", [("id", "ID", ""), ("forecast_date", "DATE", "NOT NULL"), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                         ("ticker", "TEXT", "NOT NULL"), ("model_name", "TEXT", "NOT NULL"), ("model_version", "TEXT", ""),
                         ("horizon", "INT", ""), ("exp_return", "REAL", ""), ("prob_up", "REAL", ""),
                         ("prob_bullish", "REAL", ""), ("prob_bearish", "REAL", ""), ("vol", "REAL", ""),
                         ("q10", "REAL", ""), ("q50", "REAL", ""), ("q90", "REAL", ""), ("rank_score", "REAL", ""),
                         ("run_id", "TEXT", ""),
                         ("created_at", "TS", "")],
     [["forecast_date", "stock_id", "model_name"]], [["forecast_date"], ["model_name", "forecast_date"]]),
    # Tanggal rilis laba (SEC 8-K Item 2.02) — point-in-time; ±4 baris/emiten/tahun (≈ 0,5 MB/tahun untuk 900 emiten).
    ("earnings_events", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                         ("accession", "TEXT", "NOT NULL"), ("filing_date", "DATE", ""), ("accepted_at", "TEXT", ""),
                         ("reaction_date", "DATE", "NOT NULL"), ("form", "TEXT", ""), ("source", "TEXT", ""),
                         ("retrieved_at", "TS", "")],
     [["stock_id", "accession"]], [["reaction_date"]]),
    # ---- Alpha Vantage (v9). Pemakaian kuota (jendela 24 jam bergulir, lintas workflow).
    ("api_usage", [("id", "ID", ""), ("provider", "TEXT", "NOT NULL"), ("called_at", "TS", "NOT NULL"),
                   ("function", "TEXT", ""), ("symbol", "TEXT", ""), ("status", "TEXT", "")],
     [], [["provider", "called_at"], ["function", "symbol"]]),
    # EPS aktual vs estimasi analis per kuartal (±4 baris/emiten/tahun; riwayat penuh dari 1 request).
    ("earnings_surprises", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                            ("fiscal_date_ending", "DATE", "NOT NULL"), ("reported_date", "DATE", "NOT NULL"),
                            ("report_time", "TEXT", ""), ("reported_eps", "REAL", ""), ("estimated_eps", "REAL", ""),
                            ("surprise", "REAL", ""), ("surprise_pct", "REAL", ""), ("source", "TEXT", ""),
                            ("retrieved_at", "TS", "")],
     [["stock_id", "fiscal_date_ending"]], [["reported_date"]]),
    # Kalender rilis laba — disimpan sebagai LOG PERUBAHAN (baris baru hanya bila tanggal/estimasi berubah), sehingga
    # riwayat revisi konsensus EPS terkumpul tanpa menulis ±900 baris per hari.
    ("earnings_calendar", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                           ("fiscal_date_ending", "DATE", "NOT NULL"), ("report_date", "DATE", "NOT NULL"),
                           ("estimate", "REAL", ""), ("time_of_day", "TEXT", ""), ("as_of", "DATE", "NOT NULL"),
                           ("retrieved_at", "TS", "")],
     [["stock_id", "fiscal_date_ending", "as_of"]], [["report_date"], ["as_of"]]),
    ("macro_series", [("id", "ID", ""), ("series", "TEXT", "NOT NULL"), ("date", "DATE", "NOT NULL"),
                      ("value", "REAL", ""), ("source", "TEXT", ""), ("retrieved_at", "TS", "")],
     [["series", "date"]], [["series", "date"]]),
    ("news_items", [("id", "ID", ""), ("stock_id", "INT", "NOT NULL REFERENCES stocks(id)"),
                    ("published_at", "TS", "NOT NULL"), ("url_hash", "TEXT", "NOT NULL"), ("headline", "TEXT", ""),
                    ("source", "TEXT", ""), ("sentiment", "REAL", ""), ("relevance", "REAL", ""),
                    ("retrieved_at", "TS", "")],
     [["stock_id", "url_hash"]], [["published_at"]]),
    ("schema_migrations", [("version", "INT", "PRIMARY KEY"), ("applied_at", "TS", "")], [], []),
]

TABLE_NAMES = [t[0] for t in TABLES]


def render_ddl(dialect: str, part: str = "all") -> list[str]:
    """part: tables | indexes | all. Index dibuat setelah kolom baru ditambahkan (migrasi DB lama)."""
    types = TYPES[dialect]
    tables, indexes = [], []
    for name, cols, uniques, index_cols in TABLES:
        defs = [f"{c} {types[t]} {cons}".strip() for c, t, cons in cols]
        for u in uniques:
            defs.append(f"UNIQUE ({', '.join(u)})")
        tables.append(f"CREATE TABLE IF NOT EXISTS {name} (\n  " + ",\n  ".join(defs) + "\n)")
        for idx in index_cols:
            indexes.append(f"CREATE INDEX IF NOT EXISTS ix_{name}_{'_'.join(idx)} ON {name} ({', '.join(idx)})")
    return {"tables": tables, "indexes": indexes, "all": tables + indexes}[part]


class SchemaConflictError(RuntimeError):
    pass


def column_info(db: Database, table: str) -> dict[str, dict]:
    """{kolom: {"notnull": bool, "has_default": bool}} untuk tabel yang sudah ada (kosong bila belum ada)."""
    if db.dialect == "sqlite":
        return {r[1]: {"notnull": bool(r[3]) and not r[5], "has_default": r[4] is not None or bool(r[5])}
                for r in db.query(f"PRAGMA table_info({table})")}
    rows = db.query("SELECT column_name, is_nullable, column_default FROM information_schema.columns "
                    "WHERE table_name = ? AND table_schema = current_schema()", (table,))
    return {r[0]: {"notnull": r[1] == "NO", "has_default": r[2] is not None} for r in rows}


def check_compatibility(db: Database) -> list[str]:
    """Deteksi tabel bernama sama yang dibuat program LAIN (kolom wajib yang tidak dikenal proyek ini).

    Tanpa pemeriksaan ini, bentrokan baru muncul sebagai error NOT NULL di tengah pipeline.
    """
    problems = []
    for name, cols, _, _ in TABLES:
        info = column_info(db, name)
        if not info:
            continue
        ours = {c: ("NOT NULL" in cons or "PRIMARY KEY" in cons or t == "ID") for c, t, cons in cols}
        for col, meta in info.items():
            if not meta["notnull"] or meta["has_default"]:
                continue
            if col not in ours:
                problems.append(f"{name}.{col} (kolom wajib yang tidak dikenal proyek ini)")
            elif not ours[col]:
                problems.append(f"{name}.{col} (NOT NULL, seharusnya boleh kosong)")
    return problems


def existing_columns(db: Database, table: str) -> set[str]:
    if db.dialect == "sqlite":
        return {r[1] for r in db.query(f"PRAGMA table_info({table})")}
    return {r[0] for r in db.query("SELECT column_name FROM information_schema.columns WHERE table_name = ? "
                                   "AND table_schema = current_schema()", (table,))}


def add_missing_columns(db: Database) -> list[str]:
    """Migrasi additive: kolom baru di TABLES ditambahkan ke tabel lama (data tidak disentuh)."""
    types, added = TYPES[db.dialect], []
    for name, cols, _, _ in TABLES:
        have = existing_columns(db, name)
        for c, t, _cons in cols:
            if c not in have and t != "ID":
                db.execute(f"ALTER TABLE {name} ADD COLUMN {c} {types[t]}")
                added.append(f"{name}.{c}")
    return added


def migrate(db: Database) -> int:
    """Buat tabel yang belum ada + tambahkan kolom baru. Aman dijalankan berulang (setiap run daily memanggilnya)."""
    conflicts = check_compatibility(db)
    if conflicts:
        hint = ("hapus file database lokal (data/local.db) agar dibuat ulang" if db.dialect == "sqlite" else
                "pakai service/database Aiven baru yang kosong khusus proyek ini, atau hapus tabel lama tersebut")
        raise SchemaConflictError(
            "Database sudah berisi tabel dari program lain yang tidak cocok dengan proyek ini: "
            + "; ".join(conflicts[:8]) + (f" (+{len(conflicts) - 8} lagi)" if len(conflicts) > 8 else "")
            + f". Solusi: {hint}.")
    db.executescript(render_ddl(db.dialect, "tables"))
    add_missing_columns(db)
    db.executescript(render_ddl(db.dialect, "indexes"))
    current = db.scalar("SELECT MAX(version) FROM schema_migrations") or 0
    if current < SCHEMA_VERSION:  # catat versi; perubahan skema sendiri sudah additive di atas
        from datetime import datetime, timezone
        db.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                   (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return SCHEMA_VERSION


def postgres_sql_file() -> str:
    head = ("-- Skema PostgreSQL (Aiven / PostgreSQL apa pun). Dihasilkan oleh `python main.py db-schema`.\n"
            "-- Opsional: jalankan di SQL Editor provider. `python main.py setup` juga membuatnya otomatis.\n\n")
    return head + ";\n\n".join(render_ddl("postgres")) + ";\n"
