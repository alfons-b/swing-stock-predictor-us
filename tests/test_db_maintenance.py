"""Retensi & pemeliharaan database (kuota storage paket gratis → penulisan diblokir/read-only bila terlampaui)."""
import unittest

import pandas as pd

from app.config import ConfigError, validate_config
from app.database.db import Database
from app.database.maintenance import run_maintenance, size_status
from app.database.repository import Repository
from app.database.schema import TABLE_NAMES, migrate
from tests.helpers import db_urls, make_cfg

TODAY = pd.Timestamp("2026-10-09")


def seeded(url):
    db = Database.from_url(url)
    if db.dialect == "postgres":
        db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" for t in reversed(TABLE_NAMES)])
    migrate(db)
    r = Repository(db)
    r.upsert_stocks(pd.DataFrame({"ticker": ["AAAA", "BBBB"], "is_active": [True, True]}))
    sid = r.stock_ids()["AAAA"]
    dates = pd.bdate_range("2017-01-02", "2026-10-08")[::20]                       # ±9,8 tahun harga
    r.upsert_prices(pd.DataFrame({"stock_id": sid, "date": dates, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                                  "volume": 1.0, "value": 1.0, "is_adjusted": True, "source": "t"}))
    preds = []
    for d, dec in [("2025-01-06", "BUY"), ("2025-01-06", "WAIT"), ("2025-01-07", "AVOID"), ("2025-01-08", "WATCHLIST"),
                   ("2026-09-01", "WAIT"), ("2026-09-01", "BUY")]:
        preds.append({"prediction_date": d, "stock_id": sid if dec != "AVOID" else r.stock_ids()["BBBB"], "ticker": "AAAA",
                      "model_version": f"m-{dec}", "decision": dec})
    r.save_predictions(pd.DataFrame(preds))
    ids = [x[0] for x in db.query("SELECT id FROM predictions")]
    r.save_evaluations(pd.DataFrame({"prediction_id": ids, "outcome": "TIME_EXIT"}))
    db.upsert("reports", [{"report_date": "2026-01-05", "kind": "html", "filename": "old.html", "content": b"x" * 100},
                          {"report_date": "2026-10-08", "kind": "html", "filename": "new.html", "content": b"y"}],
              keys=["report_date", "kind", "filename"])
    r.write_logs([{"run_id": "r", "ts": "2026-01-01T00:00:00+00:00", "level": "INFO", "logger": "t", "message": "old"},
                  {"run_id": "r", "ts": "2026-10-08T00:00:00+00:00", "level": "INFO", "logger": "t", "message": "new"}])
    models = [{"version": f"model_v{i:03d}", "status": "RETIRED", "created_at": f"2026-0{i}-01T00:00:00+00:00",
               "artifact": b"m" * 50} for i in range(1, 7)]
    models[0]["status"] = "ACTIVE"                                                   # yang aktif justru yang tertua
    db.upsert("model_versions", models, keys=["version"])
    vd = pd.bdate_range("2026-07-01", "2026-10-08")                               # valuasi harian ±3 bulan
    db.upsert("valuation_results", pd.DataFrame({"stock_id": sid, "as_of_date": vd.strftime("%Y-%m-%d"),
                                                 "valuation_status": "INSUFFICIENT_DATA"}), keys=["stock_id", "as_of_date"])
    db.upsert("accumulation_signals", pd.DataFrame({"stock_id": sid, "date": ["2025-01-06", "2026-10-01"],
                                                    "status": "MODERATE_ACCUMULATION_SIGNAL"}), keys=["stock_id", "date"])
    db.upsert("financial_statements", [{"stock_id": sid, "period_end": "2010-12-31", "period_type": "FY",
                                        "first_known_date": "2011-03-31", "source": "t", "version": 1}],
              keys=["stock_id", "period_end", "period_type", "source", "version"])
    return db


class TestRetention(unittest.TestCase):
    def test_dry_run_changes_nothing_then_retention_applies(self):
        for url in db_urls():
            with self.subTest(url=url.split("@")[-1]):
                db = seeded(url)
                cfg = make_cfg(db_url=url)
                n_px = db.scalar("SELECT COUNT(*) FROM price_history")
                out = run_maintenance(cfg, db, TODAY, dry_run=True)
                self.assertGreater(sum(s["rows"] for s in out["steps"]), 0)
                self.assertEqual(db.scalar("SELECT COUNT(*) FROM price_history"), n_px)

                run_maintenance(cfg, db, TODAY)
                left = {r[0] for r in db.query("SELECT decision || '@' || CAST(prediction_date AS TEXT) FROM predictions")}
                self.assertEqual(left, {"BUY@2025-01-06", "WATCHLIST@2025-01-08", "WAIT@2026-09-01", "BUY@2026-09-01"})
                self.assertEqual(db.scalar("SELECT COUNT(*) FROM prediction_evaluations"), 4)  # evaluasi ikut terhapus
                self.assertEqual(str(db.scalar("SELECT MIN(date) FROM price_history"))[:4], "2020")  # 6 tahun
                self.assertEqual([r[0] for r in db.query("SELECT filename FROM reports")], ["new.html"])
                self.assertEqual([r[0] for r in db.query("SELECT message FROM system_logs")], ["new"])
                kept = {r[0] for r in db.query("SELECT version FROM model_versions WHERE artifact IS NOT NULL")}
                self.assertEqual(kept, {"model_v001", "model_v004", "model_v005", "model_v006"})   # ACTIVE + 3 terbaru
                self.assertEqual(db.scalar("SELECT COUNT(*) FROM model_versions"), 6)  # registry tetap utuh
                vdates = sorted(str(r[0])[:10] for r in db.query("SELECT as_of_date FROM valuation_results"))
                old_v = [d for d in vdates if d < "2026-09-25"]                     # > 14 hari: hanya akhir bulan
                self.assertEqual(old_v, ["2026-07-31", "2026-08-31"])
                self.assertTrue(all(d >= "2026-09-25" for d in vdates[2:]))
                self.assertEqual([str(r[0])[:10] for r in db.query("SELECT date FROM accumulation_signals")], ["2026-10-01"])
                self.assertEqual(db.scalar("SELECT COUNT(*) FROM financial_statements"), 1)   # laporan tidak dihapus
                again = run_maintenance(cfg, db, TODAY, full=True)                    # idempoten + VACUUM FULL jalan
                self.assertEqual(sum(s["rows"] for s in again["steps"]), 0)
                db.close()

    def test_config_rejects_retention_shorter_than_history(self):
        cfg = make_cfg()
        cfg["retention"]["price_history_years"] = 4
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_size_status_thresholds(self):
        db = seeded("sqlite:///:memory:")
        cfg = make_cfg()
        size_mb = size_status(cfg, db)["bytes"] / 1e6
        cfg["database"]["size_limit_mb"] = size_mb / 0.9
        st = size_status(cfg, db)
        self.assertEqual((st["status"], st["maintenance_due"]), ("WARN", True))
        cfg["database"]["size_limit_mb"] = size_mb * 10
        self.assertEqual(size_status(cfg, db)["status"], "OK")


class TestReadOnlyRecovery(unittest.TestCase):
    def test_daily_recovers_from_supabase_read_only(self):
        pg = [u for u in db_urls() if u.startswith("postgres")]
        if not pg:
            self.skipTest("butuh TEST_POSTGRES_URL")
        from app.pipeline.context import AppContext
        from app.pipeline.jobs import PipelineAbort, ensure_writable
        seeded(pg[0]).close()
        cfg = make_cfg(db_url=pg[0], as_of=str(TODAY.date()))
        ctx = AppContext.create(cfg)
        ctx.db.execute("SET default_transaction_read_only = on")                    # meniru Supabase read-only
        with self.assertRaises(Exception):
            ctx.db.execute("INSERT INTO system_logs (run_id, message) VALUES ('x', 'y')")
        ctx.db.conn.rollback()
        ensure_writable(ctx)                                                         # retensi + VACUUM FULL
        ctx.db.execute("INSERT INTO system_logs (run_id, message) VALUES ('x', 'y')")  # bisa menulis lagi
        ctx.db.execute("SET default_transaction_read_only = on")
        ctx.cfg["database"]["size_limit_mb"] = 0.001                                 # masih di atas batas → berhenti jelas
        with self.assertRaises(PipelineAbort):
            ensure_writable(ctx)
        ctx.close()


if __name__ == "__main__":
    unittest.main()
