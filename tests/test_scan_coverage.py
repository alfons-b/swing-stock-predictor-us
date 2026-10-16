"""Cakupan pemindaian seluruh universe: setiap emiten aktif punya tepat satu status + alasan, dan hanya SUCCESS
yang boleh menjadi BUY (data tidak valid tidak menghasilkan sinyal palsu)."""
import unittest

import pandas as pd

from app.scanner.coverage import (STATUSES, apply_coverage_gate, build_coverage, coverage_csv, coverage_markdown,
                                  coverage_summary)
from tests.helpers import make_cfg, make_ctx

D = pd.Timestamp("2024-03-15")


def _cfg():
    cfg = make_cfg()
    cfg["quality"]["min_history_days"] = 120
    cfg["quality"]["min_price"] = 5
    return cfg


def _day_row(t, **over):
    base = {"ticker": t, "date": D, "close": 100.0, "atr_pct": 0.03, "ema20": 99.0, "ema50": 98.0, "rsi14": 55.0,
            "adx": 25.0, "volume_ratio": 1.2, "avg_value20": 5e7, "is_liquid": True, "is_suspended": False,
            "days_listed": 300}
    base.update(over)
    return base


class TestBuildCoverage(unittest.TestCase):
    def setUp(self):
        tick = ["OKAY", "PART", "YUNG", "PERR", "SUSP", "ILLQ", "PENY", "FCAB", "STAL", "NONE", "DROP", "GONE"]
        self.uni = pd.DataFrame({"ticker": tick, "is_active": [True] * 11 + [False],
                                 "board": ["S&P 500"] * 7 + ["OTC"] + ["S&P 400"] * 4})
        last = {t: D for t in tick}
        last.update({"STAL": D - pd.Timedelta(days=20), "PERR": D - pd.Timedelta(days=1), "NONE": pd.NaT})
        self.latest = pd.DataFrame({"ticker": tick, "last_date": [last[t] for t in tick],
                                    "n_rows": [0 if t == "NONE" else 400 for t in tick]})
        self.day = pd.DataFrame([
            _day_row("OKAY"), _day_row("PART", adx=float("nan")), _day_row("YUNG", days_listed=40),
            _day_row("SUSP", is_suspended=True), _day_row("ILLQ", is_liquid=False), _day_row("PENY", close=3.0),
            _day_row("FCAB"), _day_row("GONE")])
        self.cov = build_coverage(self.uni, self.latest, self.day, D, _cfg(),
                                  provider_failed={"PERR": "HTTPError 503"}, provider_no_data=["STAL"])
        self.st = dict(zip(self.cov["ticker"], self.cov["status"]))
        self.reason = dict(zip(self.cov["ticker"], self.cov["reason"]))

    def test_every_active_stock_has_exactly_one_status(self):
        self.assertEqual(len(self.cov), 11)                               # GONE (tidak aktif) tidak dinilai
        self.assertFalse(self.cov["ticker"].duplicated().any())
        self.assertTrue(self.cov["status"].isin(STATUSES).all())
        self.assertTrue((self.cov.loc[self.cov["status"] != "SUCCESS", "reason"].str.len() > 0).all())

    def test_statuses(self):
        expect = {"OKAY": "SUCCESS", "PART": "PARTIAL_DATA", "YUNG": "INSUFFICIENT_DATA", "PERR": "PROVIDER_ERROR",
                  "SUSP": "EXCLUDED", "ILLQ": "EXCLUDED", "PENY": "EXCLUDED", "FCAB": "EXCLUDED",
                  "STAL": "STALE_DATA", "NONE": "INSUFFICIENT_DATA", "DROP": "PARTIAL_DATA"}
        self.assertEqual(self.st, expect)

    def test_reasons_are_specific(self):
        self.assertIn("adx", self.reason["PART"])
        self.assertIn("HTTPError", self.reason["PERR"])
        self.assertIn("OTC", self.reason["FCAB"])
        self.assertIn("suspensi", self.reason["SUSP"])
        self.assertIn(str((D - pd.Timedelta(days=20)).date()), self.reason["STAL"])
        self.assertIn("provider tidak mengembalikan", self.reason["STAL"])

    def test_excluded_boards_configurable(self):
        cfg = _cfg()
        cfg["coverage"] = {"excluded_boards": []}
        cov = build_coverage(self.uni, self.latest, self.day, D, cfg)
        self.assertEqual(dict(zip(cov["ticker"], cov["status"]))["FCAB"], "SUCCESS")

    def test_summary_counts_add_up(self):
        s = coverage_summary(self.cov)
        self.assertEqual(s["total_universe"], 11)
        self.assertEqual(sum(s["status_counts"].values()), 11)
        self.assertEqual(s["processed"], 7)                               # baris hari scan yang dianalisis
        self.assertEqual(s["signal_eligible"], 1)
        self.assertEqual(s["failed"], 1)
        self.assertEqual(s["excluded"], 4)
        self.assertAlmostEqual(s["coverage_pct"], round(100 * 7 / 11, 1))
        md = "\n".join(coverage_markdown(s))
        self.assertIn("PROVIDER_ERROR | 1", md)
        csv = coverage_csv(self.cov).decode()
        self.assertTrue(csv.startswith("scan_date,ticker,status,reason"))


class TestCoverageGate(unittest.TestCase):
    def test_only_success_can_be_buy(self):
        sig = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "decision": ["BUY", "BUY", "WATCHLIST", "BUY"],
                            "reject_reasons": [[], [], ["x"], []]})
        cov = pd.DataFrame({"ticker": ["A", "B", "C"], "status": ["SUCCESS", "EXCLUDED", "STALE_DATA"]})
        out = apply_coverage_gate(sig, cov)
        self.assertEqual(out["decision"].tolist(), ["BUY", "WAIT", "WATCHLIST", "WAIT"])
        self.assertIn("coverage_excluded", out["reject_reasons"].iloc[1])
        self.assertIn("coverage_not_in_universe", out["reject_reasons"].iloc[3])
        self.assertEqual(sig["decision"].tolist(), ["BUY", "BUY", "WATCHLIST", "BUY"])   # input tidak diubah

    def test_no_coverage_is_noop(self):
        sig = pd.DataFrame({"ticker": ["A"], "decision": ["BUY"], "reject_reasons": [[]]})
        self.assertIs(apply_coverage_gate(sig, None), sig)


class TestCoverageEndToEnd(unittest.TestCase):
    """Setup → manipulasi data nyata di DB → daily: status tersimpan, laporan dibuat, tidak ada BUY non-SUCCESS."""

    @classmethod
    def setUpClass(cls):
        from app.pipeline.jobs import job_daily, job_setup
        cls.cfg = make_cfg(as_of="2020-06-30")
        job_setup(make_ctx(cls.cfg), skip_backtest=True)
        ctx = make_ctx(cls.cfg)
        tick = ctx.repo.stocks(active_only=True)["ticker"].tolist()
        cls.stale, cls.fcab = tick[0], tick[1]
        ids = ctx.repo.stock_ids()
        # emiten tanpa transaksi 10 hari terakhir → STALE_DATA
        ctx.db.execute("DELETE FROM price_history WHERE stock_id = ? AND date > ?", (ids[cls.stale], "2020-06-16"))
        # saham OTC → EXCLUDED
        ctx.db.execute("UPDATE stocks SET board = ? WHERE ticker = ?", ("OTC", cls.fcab))
        # emiten baru di universe tanpa histori → INSUFFICIENT_DATA
        ctx.repo.upsert_stocks(pd.DataFrame({"ticker": ["ZNEW"], "name": ["Baru"], "is_active": [True]}))
        ctx.close()
        cls.ctx = make_ctx(cls.cfg)
        cls.summary = job_daily(cls.ctx, skip_ingest=True)

    def test_coverage_persisted_for_whole_universe(self):
        cov = self.ctx.repo.coverage()
        n_active = int(self.ctx.db.scalar("SELECT COUNT(*) FROM stocks WHERE is_active = ?", (True,)))
        self.assertEqual(len(cov), n_active)
        st = dict(zip(cov["ticker"], cov["status"]))
        self.assertEqual(st[self.stale], "STALE_DATA")
        self.assertEqual(st[self.fcab], "EXCLUDED")
        self.assertEqual(st["ZNEW"], "INSUFFICIENT_DATA")
        self.assertEqual(self.summary["coverage"]["total_universe"], n_active)

    def test_no_signal_for_non_success(self):
        cov = self.ctx.repo.coverage()
        bad = set(cov.loc[cov["status"] != "SUCCESS", "ticker"])
        d = self.summary["date"]
        buys = {r[0] for r in self.ctx.db.query("SELECT ticker FROM predictions WHERE prediction_date = ? AND decision = ?",
                                                (d, "BUY"))}
        self.assertFalse(buys & bad)
        ranked = {r["ticker"] for r in self.summary["recommendations"]}
        self.assertFalse(ranked & bad)
        self.assertNotIn(self.stale, {r[0] for r in self.ctx.db.query(
            "SELECT ticker FROM predictions WHERE prediction_date = ?", (d,))})   # tidak ada skor palsu untuk data basi

    def test_reports_and_run_summary(self):
        kinds = {r[0] for r in self.ctx.db.query("SELECT kind FROM reports WHERE report_date = ?", (self.summary["date"],))}
        self.assertIn("coverage_csv", kinds)
        md = self.ctx.reports.latest("markdown")[1].decode()
        self.assertIn("Cakupan pemindaian", md)
        last = self.ctx.repo.last_run("daily")
        self.assertIn("coverage", str(last.get("summary")))

    def test_rerun_is_idempotent(self):
        from app.pipeline.jobs import job_daily
        n0 = int(self.ctx.db.scalar("SELECT COUNT(*) FROM scan_coverage"))
        job_daily(make_ctx(self.cfg), skip_ingest=True)
        self.assertEqual(int(self.ctx.db.scalar("SELECT COUNT(*) FROM scan_coverage")), n0)


if __name__ == "__main__":
    unittest.main()
