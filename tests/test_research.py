"""Integrasi riset end-to-end di DATA CONTOH SINTETIS: laporan emiten, ranking, batas konsentrasi, evaluasi modul.

Tes ini memeriksa alur & aturan (label, PIT, tidak ada angka karangan) — hasil numeriknya BUKAN bukti kinerja.
"""
import unittest

import numpy as np
import pandas as pd

from app.pipeline.jobs import job_daily, job_setup
from app.research.evaluation import VARIANTS, apply_variant, evaluate_modules, format_evaluation
from app.research.integrated_scoring import assemble, composite, rank_all, value_decision
from app.research.report import build_research, stock_report
from app.risk.concentration import apply_concentration_limits
from app.valuation.financial_statement_normalizer import CANONICAL, Statement
from app.valuation.fundamental_provider import store_statements
from tests.helpers import make_cfg, make_ctx


def _fixture_statements(tickers):
    """Laporan FIKTIF untuk emiten sampel (tahun 2016–2019, dipublikasikan 90 hari setelah tutup buku)."""
    out = []
    for i, t in enumerate(tickers):
        for k, y in enumerate(range(2015, 2020)):
            ni = (50.0 + 7 * i) * 1e9 * (1.04 + 0.01 * (i % 3)) ** k
            it = {c: None for c in CANONICAL}
            it.update(revenue=ni * 8, net_income=ni, operating_income=ni * 1.3, ebit=ni * 1.3, ebitda=ni * 1.7,
                      total_equity=ni * 7, total_assets=ni * 14, total_liabilities=ni * 7, total_debt=ni * 2,
                      cash=ni, operating_cash_flow=ni * 1.2, capex=ni * 0.3, free_cash_flow=ni * 0.9,
                      dividends_paid=ni * 0.3, shares_outstanding=1e9, interest_expense=ni * 0.1,
                      current_assets=ni * 3, current_liabilities=ni * 2)
            out.append(Statement(t, pd.Timestamp(f"{y}-12-31"), "FY", "USD", it, "csv_fundamentals", "fixture",
                                 publication_date=pd.Timestamp(f"{y + 1}-03-31")))
    return out


class ResearchIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_cfg(as_of="2020-06-30")
        job_setup(make_ctx(cls.cfg), skip_backtest=True)
        ctx = make_ctx(cls.cfg)
        tickers = ctx.repo.stocks()["ticker"].tolist()
        store_statements(ctx.repo, cls.cfg, _fixture_statements(tickers), pd.Timestamp("2020-06-30"), True)
        c = dict(cls.cfg)
        c["_as_of"] = "2020-07-06"
        cls.summary = job_daily(make_ctx(c))
        cls.ctx = make_ctx(c)
        cls.tickers = ctx.repo.stocks(active_only=True)["ticker"].tolist()

    def test_daily_values_with_fundamentals(self):
        rs = self.summary["research"]
        self.assertGreater(rs["valuation_ok"], 0)
        v = self.ctx.db.query_df("SELECT * FROM valuation_results WHERE as_of_date = ?", ("2020-07-06",))
        self.assertEqual(len(v), len(self.tickers))
        ok = v[v["valuation_status"] != "INSUFFICIENT_DATA"]
        self.assertTrue((pd.to_datetime(ok["fundamentals_period_end"]) <= "2019-12-31").all())
        self.assertTrue((pd.to_datetime(ok["fundamentals_known_date"]) <= "2020-07-06").all())

    def test_stock_report_and_rankings(self):
        res = build_research(self.cfg, self.ctx.repo, "2020-07-06")
        t = res["table"]
        self.assertEqual(len(t), len(self.tickers))
        txt = stock_report(res, self.tickers[0])
        for part in ("STOCK RESEARCH REPORT", "VALUE decision", "SWING decision", "[VALUATION]", "[SHORT FLOW — FINRA]",
                     "[ACCUMULATION / DISTRIBUTION]", "[SKOR TERPISAH", "[RANKING", "FOREIGN_FLOW_UNAVAILABLE"):
            self.assertIn(part, txt)
        # foreign flow tidak tersedia → ranking foreign_buying kosong (bukan diisi nol)
        self.assertTrue(t["rank_foreign_buying"].isna().all())
        self.assertTrue(t["foreign_flow_score"].isna().all())

    def test_dashboard_data(self):
        from app.research import dashboard_data as dd
        cov = dd.valuation_coverage(self.ctx.repo)
        self.assertEqual(cov["total"], len(self.tickers))
        scr = dd.undervalued_screener(self.ctx.repo, -9, ("DEEP_VALUE", "UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"), False)
        self.assertEqual(len(scr), cov["total"] - cov["by_status"].get("INSUFFICIENT_DATA", 0))
        self.assertEqual(dd.foreign_flow_overview(self.ctx.repo)["status"], "FOREIGN_FLOW_UNAVAILABLE")
        self.assertEqual(len(dd.accumulation_table(self.ctx.repo)), len(self.tickers))
        t = dd.integrated_table(self.ctx.repo, self.cfg)
        self.assertEqual(len(t), len(self.tickers))
        rv = dd.ranking_view(t, "swing_setup", 5)
        self.assertLessEqual(len(rv), 5)
        self.assertTrue(dd.ranking_view(t, "foreign_buying").empty)
        self.assertIn("fundamental", set(dd.source_status(self.ctx.repo)["category"]))

    def test_evaluate_modules_structure(self):
        res = evaluate_modules(self.cfg, self.ctx.repo, save=True)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["pit_mode"], "PIT")
        for p in res["periods"].values():
            self.assertEqual(set(p["variants"]), set(VARIANTS))
            self.assertEqual(p["variants"]["V4_with_foreign_flow"]["status"], "INSUFFICIENT_DATA")   # tanpa histori flow
            self.assertEqual(p["variants"]["V1_swing_baseline"]["status"], "OK")
        self.assertEqual(res["factor_ic"]["foreign_flow_score"]["status"], "FOREIGN_FLOW_UNAVAILABLE")
        self.assertIn("EVALUASI MODUL", format_evaluation(res))
        self.assertEqual(self.ctx.db.scalar("SELECT COUNT(*) FROM backtest_runs WHERE kind = 'module_evaluation'"), 1)


class ScoringUnitTest(unittest.TestCase):
    def test_composite_never_fills_zero(self):
        t = pd.DataFrame({"a": [80.0, np.nan, np.nan], "b": [60.0, 40.0, np.nan]})
        c = composite(t, {"a": 0.5, "b": 0.5}, 0.5)
        self.assertAlmostEqual(c[0], 70.0)
        self.assertAlmostEqual(c[1], 40.0)                     # bobot dinormalisasi ulang, bukan (0 + 40) / 2
        self.assertTrue(np.isnan(c[2]))
        self.assertTrue(np.isnan(composite(t, {"a": 0.8, "b": 0.2}, 0.6)[1]))   # cakupan bobot 20% < 60%

    def test_value_decision_separate_from_swing(self):
        base = {"valuation_status": "UNDERVALUED", "value_trap_risk": "LOW", "business_quality_score": 70.0,
                "valuation_confidence": "MEDIUM"}
        self.assertEqual(value_decision(base), "VALUE_CANDIDATE")
        self.assertEqual(value_decision({**base, "value_trap_risk": "HIGH"}), "VALUE_TRAP_RISK")
        self.assertEqual(value_decision({**base, "valuation_confidence": "LOW"}), "VALUE_WATCH")
        self.assertEqual(value_decision({**base, "valuation_status": "OVERVALUED"}), "NOT_UNDERVALUED")
        self.assertEqual(value_decision({"valuation_status": "INSUFFICIENT_DATA"}), "INSUFFICIENT_DATA")
        sig = pd.DataFrame({"ticker": ["A"], "decision": ["AVOID"], "technical_score": [20.0], "ml_score": [10.0],
                            "liquidity_score": [50.0], "market_regime": ["BEAR"]})
        val = pd.DataFrame([{"ticker": "A", "value_score": 85.0, "quality_score": 70.0, **base}])
        t = rank_all(assemble(sig, val, None, None), make_cfg())
        self.assertEqual(t.iloc[0]["swing_decision"], "AVOID")           # valuasi tidak mengubah keputusan swing
        self.assertEqual(t.iloc[0]["value_decision"], "VALUE_CANDIDATE")

    def test_value_trap_excluded_from_value_rankings(self):
        sig = pd.DataFrame({"ticker": ["A", "B"], "decision": ["WAIT"] * 2, "technical_score": [50.0] * 2,
                            "ml_score": [50.0] * 2, "liquidity_score": [50.0] * 2, "market_regime": ["NEUTRAL"] * 2})
        val = pd.DataFrame({"ticker": ["A", "B"], "value_score": [95.0, 70.0], "quality_score": [60.0, 60.0],
                            "value_trap_risk": ["HIGH", "LOW"], "valuation_status": ["DEEP_VALUE", "UNDERVALUED"],
                            "valuation_confidence": ["HIGH", "HIGH"]})
        t = rank_all(assemble(sig, val, None, None), make_cfg()).set_index("ticker")
        self.assertTrue(np.isnan(t.loc["A", "rank_best_value"]))
        self.assertEqual(t.loc["B", "rank_best_value"], 1)

    def test_variant_filters_only_touch_buys(self):
        s = pd.DataFrame({"decision": ["BUY", "BUY", "WAIT"], "accumulation_score": [70.0, 30.0, 90.0],
                          "distribution_risk_score": [10.0, 80.0, 90.0], "rank_score": [1.0, 2.0, 3.0]})
        v3 = apply_variant("V3_with_accumulation", s)
        self.assertEqual(v3["decision"].tolist(), ["BUY", "WAIT", "WAIT"])
        v2 = apply_variant("V2_excl_distribution", s)
        self.assertEqual(v2["decision"].tolist(), ["BUY", "WAIT", "WAIT"])


class TechnicalScoreTest(unittest.TestCase):
    def test_missing_component_renormalized_not_zero(self):
        from app.strategy.scoring import technical_score
        w = {"trend": 0.5, "sector_strength": 0.2, "momentum": 0.3}
        comp = pd.DataFrame({"score_trend": [80.0, 80.0], "score_sector_strength": [np.nan, 50.0],
                             "score_momentum": [60.0, 60.0]})
        s = technical_score(comp, w)
        self.assertAlmostEqual(s[0], (0.5 * 80 + 0.3 * 60) / 0.8)           # bukan dihukum karena sektor kosong
        self.assertAlmostEqual(s[1], 0.5 * 80 + 0.2 * 50 + 0.3 * 60)
        self.assertTrue(np.isnan(technical_score(comp.assign(score_trend=np.nan), w)[0]))   # cakupan 30% < 70%


class LiveCalibrationTest(unittest.TestCase):
    def test_insufficient_then_ok(self):
        from app.database.db import Database
        from app.database.repository import Repository
        from app.database.schema import migrate
        from app.pipeline.evaluate import live_calibration
        db = Database.from_url("sqlite:///:memory:")
        migrate(db)
        repo = Repository(db)
        repo.upsert_stocks(pd.DataFrame({"ticker": ["AAAA"], "is_active": [True]}))
        sid = repo.stock_ids()["AAAA"]
        rng = np.random.default_rng(1)
        n = 400
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=20), periods=n)
        prob = rng.uniform(0.1, 0.6, n)
        repo.save_predictions(pd.DataFrame({"prediction_date": dates.strftime("%Y-%m-%d"), "stock_id": sid, "ticker": "AAAA",
                                            "model_version": "m", "decision": "WAIT", "prob_bullish": prob}))
        ids = [r[0] for r in db.query("SELECT id FROM predictions ORDER BY prediction_date")]
        self.assertEqual(live_calibration(repo, days=10_000)["status"], "INSUFFICIENT_DATA")
        actual = np.where(rng.uniform(size=n) < prob, 2, 1)                    # terkalibrasi sempurna (fixture)
        repo.save_evaluations(pd.DataFrame({"prediction_id": ids, "actual_class": actual, "outcome": "TIME_EXIT"}))
        cal = live_calibration(repo, days=10_000)
        self.assertEqual(cal["status"], "OK")
        self.assertLess(cal["ece_bullish"], 0.08)


class ConcentrationTest(unittest.TestCase):
    def test_sector_cap_and_correlation(self):
        cfg = make_cfg()
        cfg["research_scoring"].update(max_per_sector_buy=2, max_pair_correlation=0.8, correlation_window=60)
        d = pd.Timestamp("2024-06-28")
        sig = pd.DataFrame({"date": d, "ticker": ["A", "B", "C", "D", "E"], "decision": "BUY",
                            "sector": ["Bank", "Bank", "Bank", "Energy", "Unknown"], "rank_score": [90, 80, 70, 60, 50],
                            "reject_reasons": [[] for _ in range(5)]})
        rng = np.random.default_rng(0)
        dates = pd.bdate_range(end=d, periods=80)
        base = rng.normal(0, 0.01, 80)
        px = []
        for t, r in {"A": rng.normal(0, 0.01, 80), "B": rng.normal(0, 0.01, 80), "C": rng.normal(0, 0.01, 80),
                     "D": base, "E": base + rng.normal(0, 0.001, 80)}.items():
            px.append(pd.DataFrame({"date": dates, "ticker": t, "close": 100 * np.cumprod(1 + r)}))
        out = apply_concentration_limits(sig, cfg, pd.concat(px)).set_index("ticker")
        self.assertEqual(out.loc["C", "decision"], "WATCHLIST")
        self.assertIn("sector_concentration", out.loc["C", "reject_reasons"])
        self.assertEqual(out.loc["E", "decision"], "WATCHLIST")          # korelasi ~1 dengan D (peringkat lebih tinggi)
        self.assertIn("high_correlation_with_higher_ranked_buy", out.loc["E", "reject_reasons"])
        self.assertEqual(out.loc[["A", "B", "D"], "decision"].tolist(), ["BUY"] * 3)


if __name__ == "__main__":
    unittest.main()
