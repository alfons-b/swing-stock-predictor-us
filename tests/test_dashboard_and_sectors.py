"""Perbaikan dari audit dashboard produksi (13 Okt 2026)."""
import os
import unittest

import pandas as pd

from app.reporting import dashboard_helpers as H


class _DB:
    def __init__(self, provider):
        self.provider = provider

    def describe(self):
        return f"postgres[{self.provider}]:host:5432/db"


class TestDatabaseBanner(unittest.TestCase):
    def test_supabase_and_neon_block_dashboard(self):
        for prov, env in (("supabase", "ALLOW_SUPABASE_DATABASE"), ("neon", "ALLOW_NEON_DATABASE")):
            os.environ.pop(env, None)
            lvl, msg = H.database_banner(_DB(prov))
            self.assertEqual(lvl, "error")
            self.assertIn("Aiven", msg)

    def test_aiven_ok_and_caption_has_no_host(self):
        self.assertIsNone(H.database_banner(_DB("aiven")))
        cap = H.connection_caption(_DB("aiven"), "2026-10-12")
        self.assertIn("postgres[aiven]", cap)
        self.assertNotIn("host", cap)


WF = {"start": "2026-04-10", "end": "2026-10-09", "total_return": -0.052, "max_drawdown": -0.085, "sharpe": -1.01,
      "number_of_trades": 44}
MODULES = {"periods": {"test": {"variants": {"V1_swing_baseline": {"status": "OK", "data_coverage": 1,
                                                                     "metrics": {"total_return": -0.108, "sharpe": -2.0,
                                                                                 "number_of_trades": 83}},
                                             "V4_with_foreign_flow": {"status": "INSUFFICIENT_DATA", "data_coverage": 0}}}}}


class TestBacktestViews(unittest.TestCase):
    def test_kpis_only_for_walk_forward_format(self):
        self.assertEqual(H.backtest_kpis(WF)["number_of_trades"], 44)
        self.assertIsNone(H.backtest_kpis(MODULES))         # dulu tampil +0,0% / 0 trade
        self.assertIsNone(H.backtest_kpis(None))

    def test_module_variant_table(self):
        t = H.module_variant_table(MODULES)
        self.assertEqual(len(t), 2)
        v1 = t[t["varian"] == "V1_swing_baseline"].iloc[0]
        self.assertAlmostEqual(v1["total_return"], -0.108)
        self.assertTrue(pd.isna(t[t["varian"] == "V4_with_foreign_flow"].iloc[0]["total_return"]))

    def test_default_run_prefers_latest_wf_test(self):
        runs = pd.DataFrame({"run_id": ["modules-2", "live-2", "wf-test-2", "wf-validation-2", "wf-test-1"]})
        self.assertEqual(H.default_backtest_run(runs), 2)
        self.assertEqual(H.default_backtest_run(pd.DataFrame({"run_id": ["modules-1"]})), 0)

    def test_verdict_is_honest_about_losses(self):
        lvl, msg = H.strategy_verdict(WF, {"ihsg_buy_hold": {"total_return": -0.183},
                                           "technical_only": {"total_return": -0.108}})
        self.assertEqual(lvl, "warning")
        self.assertIn("-5.2%", msg)
        self.assertIn("S&P 500 -18.3%", msg)
        self.assertIn("RUGI", msg)
        lvl, msg = H.strategy_verdict({**WF, "total_return": 0.03}, {})
        self.assertEqual(lvl, "info")
        self.assertIn("belum membuktikan", msg)
        self.assertIsNone(H.strategy_verdict(MODULES, {}))


class TestSplitBuys(unittest.TestCase):
    def test_buys_outside_regime_cap_are_separated(self):
        buys = pd.DataFrame({"ticker": ["RSCH", "PKPK"], "score": [73.8, 79.1]})
        ranked = pd.DataFrame({"ticker": ["PKPK"], "rank": [1]})
        inside, outside = H.split_buys(buys, ranked)
        self.assertEqual(inside["ticker"].tolist(), ["PKPK"])
        self.assertEqual(outside["ticker"].tolist(), ["RSCH"])

    def test_no_ranked(self):
        inside, outside = H.split_buys(pd.DataFrame({"ticker": ["A"]}), pd.DataFrame())
        self.assertTrue(inside.empty)
        self.assertEqual(outside["ticker"].tolist(), ["A"])


class TestMissingBusinessDays(unittest.TestCase):
    def test_holiday_gap(self):
        d = pd.to_datetime(["2026-03-16", "2026-03-17", "2026-03-23"])   # 18–20 Mar libur, 21–22 akhir pekan
        self.assertEqual(H.missing_business_days(d), ["2026-03-18", "2026-03-19", "2026-03-20"])
        self.assertEqual(H.missing_business_days([]), [])


class _Prov:
    """Provider tiruan: hanya sebagian emiten punya sektor (seperti Yahoo untuk emiten suspensi)."""
    def __init__(self, sectors):
        self.sectors, self.calls = sectors, []

    def enrich_universe(self, tickers, limit=None):
        tickers = tickers if limit is None else tickers[:limit]
        self.calls.append(list(tickers))
        rows = [{"ticker": t, "name": t, "sector": self.sectors.get(t), "subsector": None} for t in tickers]
        return pd.DataFrame(rows, columns=["ticker", "name", "sector", "subsector"])


class _Chain:
    def __init__(self, prov):
        self.providers = [prov]


class TestSectorQueue(unittest.TestCase):
    def setUp(self):
        from app.data.universe import UniverseManager
        from tests.helpers import make_cfg, make_ctx
        self.ctx = make_ctx(make_cfg())
        tick = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
        self.ctx.repo.upsert_stocks(pd.DataFrame({"ticker": tick, "is_active": True, "sector": None}))
        # AAAA & BBBB tidak punya sektor di provider (dulu: memblokir antrean selamanya)
        self.prov = _Prov({"CCCC": "Energy", "DDDD": "Industrials", "EEEE": "Healthcare"})
        self.um = UniverseManager(self.ctx.cfg, self.ctx.repo, _Chain(self.prov))

    def tearDown(self):
        self.ctx.close()

    def _sectors(self):
        st = self.ctx.repo.stocks()
        return dict(zip(st["ticker"], st["sector"]))

    def test_queue_rotates_past_tickers_without_sector(self):
        import time
        out1 = self.um._enrich_metadata(limit=2)
        self.assertEqual(self.prov.calls[0], ["AAAA", "BBBB"])
        self.assertEqual((out1["attempted"], out1["filled"]), (2, 0))
        time.sleep(1.1)                                   # sector_checked_at beresolusi detik
        out2 = self.um._enrich_metadata(limit=2)
        self.assertEqual(self.prov.calls[1], ["CCCC", "DDDD"])   # tidak mengulang AAAA/BBBB
        self.assertEqual(out2["filled"], 2)
        time.sleep(1.1)
        self.um._enrich_metadata(limit=2)
        self.assertEqual(self.prov.calls[2][0], "EEEE")
        s = self._sectors()
        self.assertEqual((s["CCCC"], s["DDDD"], s["EEEE"]), ("Energy", "Industrials", "Healthcare"))

    def test_filled_sector_survives_universe_refresh_semantics(self):
        self.um._enrich_metadata(limit=10)
        self.assertEqual(self._sectors()["EEEE"], "Healthcare")
        out = self.um._enrich_metadata(limit=10)                 # hanya AAAA/BBBB yang tersisa
        self.assertEqual(out["attempted"], 2)
        self.assertEqual(out["missing"], 2)


if __name__ == "__main__":
    unittest.main()


class TestStaleCodeDetection(unittest.TestCase):
    def _repo(self, head, ref_sha=None, packed=None):
        import tempfile
        from pathlib import Path
        root = Path(tempfile.mkdtemp())
        (root / ".git" / "refs" / "heads").mkdir(parents=True)
        (root / ".git" / "HEAD").write_text(head + "\n")
        if ref_sha:
            (root / ".git" / "refs" / "heads" / "main").write_text(ref_sha + "\n")
        if packed:
            (root / ".git" / "packed-refs").write_text(packed)
        return root

    def test_reads_loose_packed_and_detached(self):
        a, b = "a" * 40, "b" * 40
        self.assertEqual(H.git_commit(self._repo("ref: refs/heads/main", ref_sha=a)), a)
        self.assertEqual(H.git_commit(self._repo("ref: refs/heads/main", packed=f"# pack\n{b} refs/heads/main\n")), b)
        self.assertEqual(H.git_commit(self._repo(a)), a)
        from pathlib import Path
        self.assertIsNone(H.git_commit(Path("/nonexistent")))

    def test_warning_only_when_commit_changed(self):
        self.assertIsNone(H.stale_code_warning("a" * 40, "a" * 40))
        self.assertIsNone(H.stale_code_warning(None, "a" * 40))
        msg = H.stale_code_warning("b" * 40, "a" * 40)
        self.assertIn("Reboot app", msg)
        self.assertIn("bbbbbbb", msg)
