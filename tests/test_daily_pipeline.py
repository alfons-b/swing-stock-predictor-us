"""End-to-end (§70, lokal): setup → daily → rerun idempoten → catch-up → evaluasi → data stale → health."""
import json
import unittest

from app.pipeline.health import health_check
from app.pipeline.jobs import job_daily, job_setup
from tests.helpers import make_cfg, make_ctx


def ctx_at(cfg, as_of):
    c = dict(cfg)
    c["_as_of"] = as_of
    return make_ctx(c)


class TestDailyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_cfg(as_of="2020-06-30")
        cls.setup = job_setup(make_ctx(cls.cfg), skip_backtest=True)

    def test_01_setup_creates_active_model(self):
        ctx = make_ctx(self.cfg)
        self.assertIn(self.setup["train"]["decision"], ("PROMOTED", "PROMOTED_AS_BASELINE"))
        self.assertIsNotNone(ctx.models.active_version())
        runs = ctx.repo.pipeline_runs()
        self.assertTrue({"setup", "validate_data", "features", "retrain"} <= set(runs["run_type"]))
        self.assertFalse((runs["status"] == "FAILED").any())

    def test_02_daily_incremental_and_outputs(self):
        ctx = ctx_at(self.cfg, "2020-07-06")
        before = ctx.repo.price_counts()["rows"]
        s = job_daily(ctx)
        self.assertEqual(s["date"], "2020-07-06")
        self.assertEqual(s["data_freshness"]["status"], "UP_TO_DATE")
        added = ctx.repo.price_counts()["rows"] - before
        n = int(ctx.db.scalar("SELECT COUNT(*) FROM stocks WHERE is_active = ?", (True,)))
        self.assertTrue(0 < added <= 3 * n)                       # hanya 3 hari bursa baru
        self.assertEqual(ctx.repo.price_counts()["duplicates"], 0)
        n_pred = ctx.db.scalar("SELECT COUNT(*) FROM predictions WHERE prediction_date = ?", ("2020-07-06",))
        self.assertGreater(n_pred, 5)
        self.assertEqual(ctx.db.scalar("SELECT COUNT(DISTINCT model_version) FROM predictions"), 1)
        kinds = {r[0] for r in ctx.db.query("SELECT kind FROM reports WHERE report_date = ?", ("2020-07-06",))}
        self.assertTrue({"predictions_csv", "html", "markdown"} <= kinds)
        html = ctx.reports.latest("html")[1].decode()
        self.assertIn("Sector ranking", html)
        self.assertIn("DATA CONTOH SINTETIS", html)                # data sintetis selalu ditandai
        last = ctx.repo.last_run("daily")
        self.assertIn(last["status"], ("SUCCESS", "PARTIAL_SUCCESS"))
        self.assertGreater(ctx.db.scalar("SELECT COUNT(*) FROM system_logs WHERE run_id = ?", (last["run_id"],)), 0)
        # modul riset ikut berjalan; tanpa sumber fundamental/flow → label, bukan angka karangan
        rs = s["research"]
        self.assertTrue(all(v == "OK" for v in rs["status"].values()), rs["status"])
        self.assertEqual(rs["foreign_flow_status"], "FOREIGN_FLOW_UNAVAILABLE")
        self.assertEqual(rs["valuation_ok"], 0)
        st = ctx.db.query_df("SELECT valuation_status, value_score, accumulation_status, foreign_flow_score "
                             "FROM predictions WHERE prediction_date = ?", ("2020-07-06",))
        self.assertTrue((st["valuation_status"] == "INSUFFICIENT_DATA").all())
        self.assertTrue(st["value_score"].isna().all() and st["foreign_flow_score"].isna().all())
        self.assertTrue(st["accumulation_status"].notna().all())
        self.assertNotIn("STRONG_ACCUMULATION_SIGNAL", set(st["accumulation_status"]))   # tanpa flow → maks MODERATE
        self.assertIn("research_csv", kinds)
        src = dict(ctx.db.query("SELECT name, status FROM data_sources WHERE category IN ('fundamental', 'foreign_flow')"))
        self.assertEqual(src.get("csv_fundamentals"), "NOT_CONFIGURED")
        self.assertLessEqual(len(s["recommendations"]), self.cfg["scoring"]["TOP_N_STOCKS"])
        for r in s["recommendations"]:
            self.assertGreaterEqual(r["risk_reward"], self.cfg["strategy"]["MIN_RISK_REWARD"])
            self.assertEqual(r["position_size"], int(r["position_size"]))     # saham utuh (tanpa lot)

    def test_03_rerun_same_day_idempotent(self):
        ctx = ctx_at(self.cfg, "2020-07-06")
        p0 = ctx.db.scalar("SELECT COUNT(*) FROM predictions")
        r0 = ctx.repo.price_counts()["rows"]
        job_daily(ctx)
        self.assertEqual(ctx.db.scalar("SELECT COUNT(*) FROM predictions"), p0)
        self.assertEqual(ctx.repo.price_counts()["rows"], r0)

    def test_04_catch_up_and_evaluate(self):
        ctx = ctx_at(self.cfg, "2020-07-24")                     # beberapa run terlewat
        s = job_daily(ctx)
        self.assertEqual(s["date"], "2020-07-24")
        ev = ctx.db.scalar("SELECT COUNT(*) FROM prediction_evaluations e JOIN predictions p ON p.id = e.prediction_id "
                           "WHERE p.prediction_date = ?", ("2020-07-06",))
        self.assertGreater(ev, 0)                                  # prediksi 6 Juli sudah lewat horizon → dievaluasi

    def test_05_stale_data_blocks_buy(self):
        ctx = ctx_at(self.cfg, "2020-08-14")
        s = job_daily(ctx, skip_ingest=True)                       # provider dianggap tidak memberi data baru
        self.assertEqual(s["data_freshness"]["status"], "STALE")
        self.assertEqual(s["recommendations"], [])
        self.assertTrue(s["headline"].startswith("NO TRADE"))

    def test_06_health(self):
        h = health_check(ctx_at(self.cfg, "2020-07-24"), check_provider=True)
        self.assertTrue(h["ok"])
        self.assertEqual(h["checks"]["ACTIVE MODEL"]["status"], "OK")
        self.assertEqual(h["checks"]["MARKET DATA PROVIDER"]["status"], "OK")
        json.dumps(h, default=str)


if __name__ == "__main__":
    unittest.main()


class TestSetupFailFast(unittest.TestCase):
    def test_short_history_aborts_before_download(self):
        """Regresi: histori terlalu pendek harus berhenti SEBELUM unduhan & training yang lama."""
        from app.pipeline.jobs import PipelineAbort
        cfg = make_cfg(as_of="2020-06-30")
        cfg["data"]["start_date"] = "2019-06-01"
        ctx = make_ctx(cfg)
        with self.assertRaises(PipelineAbort) as cm:
            job_setup(ctx, skip_backtest=True)
        self.assertIn("initial_history_years", str(cm.exception))
        self.assertEqual(ctx.repo.price_counts()["rows"], 0)
