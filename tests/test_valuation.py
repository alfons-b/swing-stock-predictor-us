"""Valuasi fundamental: normalisasi, TTM, validitas rasio, metode nilai wajar, MoS, value trap, point-in-time.

Angka laporan di sini adalah FIXTURE UJI buatan tangan (emiten fiktif) untuk memeriksa kebenaran rumus —
bukan data pasar dan bukan bukti kinerja.
"""
import json
import unittest

import numpy as np
import pandas as pd

from app.database.db import Database
from app.database.repository import Repository
from app.database.schema import TABLE_NAMES, migrate
from app.features.fundamental_sentiment import fundamental_risk_flags
from app.valuation import intrinsic_value as iv
from app.valuation.financial_statement_normalizer import CANONICAL, Statement, normalize_items, quality_checks
from app.valuation.fundamental_provider import load_statements, store_statements
from app.valuation.margin_of_safety import classify, margin_of_safety
from app.valuation.valuation_metrics import build_snapshot, fundamental_ratios, price_multiples, sector_type
from app.valuation.valuation_report import run_valuation, save_valuations
from app.valuation.valuation_scorer import quality_score, value_score
from app.valuation.value_trap_detector import detect
from tests.helpers import db_urls, make_cfg

VCFG = {"risk_free_rate": 0.07, "equity_risk_premium": 0.07, "default_beta": 1.0, "terminal_growth": 0.04,
        "max_explicit_growth": 0.15, "explicit_years": 5, "cost_of_debt_pre_tax": 0.09, "tax_rate": 0.22}


def items(**kw):
    return {**{k: None for k in CANONICAL}, **kw}


def fy_row(year, rev, ni, eq=1000.0, ocf=None, capex=20.0, div=None, debt=200.0, cash=100.0, known=None):
    it = items(revenue=rev, net_income=ni, operating_income=ni * 1.3, ebit=ni * 1.3, ebitda=ni * 1.6, total_equity=eq,
               total_assets=eq * 2, total_liabilities=eq, total_debt=debt, cash=cash, operating_cash_flow=ocf or ni * 1.1,
               capex=capex, free_cash_flow=(ocf or ni * 1.1) - capex, dividends_paid=div, shares_outstanding=100.0,
               interest_expense=10.0, current_assets=300.0, current_liabilities=200.0)
    pe = pd.Timestamp(f"{year}-12-31")
    return {"period_end": pe, "period_type": "FY", "items": it, "currency": "USD", "quality_status": "OK",
            "first_known_date": known or pe + pd.Timedelta(days=90), "usable_from": known or pe + pd.Timedelta(days=90)}


def q_row(end, rev, ni, known=None):
    pe = pd.Timestamp(end)
    it = items(revenue=rev, net_income=ni, total_equity=1000.0, operating_cash_flow=ni, capex=5.0, total_debt=200.0,
               cash=100.0, shares_outstanding=100.0)
    return {"period_end": pe, "period_type": "Q", "items": it, "currency": "USD", "quality_status": "OK",
            "first_known_date": known or pe + pd.Timedelta(days=45), "usable_from": known or pe + pd.Timedelta(days=45)}


class NormalizerTest(unittest.TestCase):
    def test_signs_and_fcf(self):
        it = normalize_items({"Total Revenue": 1000, "Operating Cash Flow": 300, "Capital Expenditure": -120,
                              "Cash Dividends Paid": -50, "Net Income": 100, "Operating Income": 150})
        self.assertEqual(it["capex"], 120)
        self.assertEqual(it["dividends_paid"], 50)
        self.assertEqual(it["free_cash_flow"], 180)
        self.assertEqual(it["ebit"], 150)
        self.assertIsNone(it["total_debt"])                      # tidak tersedia = None, bukan 0

    def test_quality_currency_and_balance(self):
        st = Statement("TEST", pd.Timestamp("2024-12-31"), "FY", "EUR",
                       items(revenue=10.0, net_income=1.0, total_equity=5.0, total_assets=20.0, total_liabilities=5.0),
                       "t", "now")
        quality_checks(st, "USD")
        self.assertEqual(st.quality_status, "CHECK")
        self.assertTrue(any("mata uang" in n for n in st.quality_notes))
        self.assertTrue(any("aset" in n for n in st.quality_notes))


class SnapshotTest(unittest.TestCase):
    def test_ttm_only_with_discrete_quarters(self):
        rows = [fy_row(2023, 400.0, 40.0)] + [q_row(d, 100.0, 10.0) for d in
                                              ("2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31")]
        rows.append(q_row("2024-03-31", 130.0, 16.0))
        snap = build_snapshot(pd.DataFrame(rows))
        self.assertTrue(snap["quarters_discrete"])
        self.assertEqual(snap["basis"], "TTM")
        self.assertAlmostEqual(snap["items"]["revenue"], 430.0)
        self.assertAlmostEqual(snap["items"]["net_income"], 46.0)

    def test_cumulative_quarters_fall_back_to_fy(self):
        # kuartal kumulatif (YTD): jumlah 4 kuartal ≠ FY → tidak boleh dijumlahkan
        rows = [fy_row(2023, 400.0, 40.0)] + [q_row(d, v, v / 10) for d, v in
                                              (("2023-03-31", 100.0), ("2023-06-30", 200.0), ("2023-09-30", 300.0),
                                               ("2023-12-31", 400.0))]
        rows.append(q_row("2024-03-31", 130.0, 13.0))
        snap = build_snapshot(pd.DataFrame(rows))
        self.assertFalse(snap["quarters_discrete"])
        self.assertEqual(snap["basis"], "FY")
        self.assertAlmostEqual(snap["items"]["revenue"], 400.0)

    def test_quarters_without_fy_insufficient(self):
        snap = build_snapshot(pd.DataFrame([q_row("2024-03-31", 100.0, 10.0)]))
        self.assertEqual(snap["basis"], "INSUFFICIENT")
        self.assertFalse(snap["flow_ok"])


class RatioValidityTest(unittest.TestCase):
    def test_loss_makes_per_invalid(self):
        snap = build_snapshot(pd.DataFrame([fy_row(2022, 400.0, 30.0), fy_row(2023, 380.0, -20.0)]))
        m, bad = price_multiples(snap, "GENERAL", 1000.0, 100.0, 1.0)
        self.assertNotIn("per", m)
        self.assertIn("per", bad)
        self.assertIn("pbv", m)
        self.assertAlmostEqual(m["pbv"], 1000.0 * 100 / 1000.0)

    def test_bank_excludes_ev_and_leverage(self):
        snap = build_snapshot(pd.DataFrame([fy_row(2022, 400.0, 30.0), fy_row(2023, 420.0, 35.0)]))
        m, bad = price_multiples(snap, "BANK", 1000.0, 100.0, 1.0)
        self.assertNotIn("ev_ebitda", m)
        r, rbad = fundamental_ratios(snap, "BANK")
        self.assertNotIn("debt_to_equity", r)
        self.assertIn("tidak relevan", rbad["debt_to_equity"])

    def test_foreign_currency_without_fx(self):
        snap = build_snapshot(pd.DataFrame([fy_row(2023, 400.0, 30.0)]))
        m, bad = price_multiples(snap, "COMMODITY", 1000.0, 100.0, None)
        self.assertEqual(m, {})
        self.assertIn("kurs", bad["*"])

    def test_sector_type(self):
        self.assertEqual(sector_type("Financials", "Banks"), "BANK")
        self.assertEqual(sector_type("Energy", "Coal"), "COMMODITY")
        self.assertEqual(sector_type("Consumer Non-Cyclicals", "Food"), "GENERAL")
        self.assertEqual(sector_type(None, None), "UNKNOWN")

    def test_bank_not_rejected_by_de_filter(self):
        df = pd.DataFrame({"fund_debt_to_equity": [6.0, 6.0], "fund_roe": [0.15, 0.15],
                           "sector": ["Financials", "Industrials"], "subsector": ["Banks", "Machinery"]})
        flags = fundamental_risk_flags(df, {"fundamental_filter": {"enabled": True, "max_debt_to_equity": 4.0}})
        self.assertEqual(flags.tolist(), [False, True])


class FairValueTest(unittest.TestCase):
    def test_dcf_matches_gordon_when_growth_equals_terminal(self):
        v = iv.dcf_per_share(100.0, 0.04, 0.10, 0.04, 1, 0.0, 1.0)
        self.assertAlmostEqual(v, 104.0 / 0.06, places=6)

    def test_dcf_sensitivity_monotonic_and_range(self):
        hist = [fy_row(y, 400.0 * 1.08 ** i, 40.0 * 1.08 ** i, capex=10.0) for i, y in enumerate(range(2019, 2024))]
        snap = build_snapshot(pd.DataFrame(hist))
        r = iv.dcf_value(snap, "GENERAL", 100.0, 1.0, 50_000.0, VCFG)
        self.assertTrue(r["valid"], r["reason"])
        self.assertLessEqual(r["low"], r["base"])
        self.assertLessEqual(r["base"], r["high"])
        sens = r["detail"]["sensitivity"]
        w = sorted({float(k.split("|")[0][5:]) for k in sens})
        g = f"{VCFG['terminal_growth']:.3f}"
        vals = [sens[f"wacc={x:.3f}|g={g}"] for x in w]
        self.assertTrue(vals[0] > vals[1] > vals[2])            # discount rate naik → nilai turun

    def test_dcf_invalid_for_bank_and_negative_fcf(self):
        snap = build_snapshot(pd.DataFrame([fy_row(y, 400.0, 40.0, capex=200.0) for y in range(2020, 2024)]))
        self.assertFalse(iv.dcf_value(snap, "BANK", 100.0, 1.0, 1e4, VCFG)["valid"])
        r = iv.dcf_value(snap, "GENERAL", 100.0, 1.0, 1e4, VCFG)
        self.assertFalse(r["valid"])
        self.assertIn("FCF", r["reason"])

    def test_justified_pb(self):
        snap = build_snapshot(pd.DataFrame([fy_row(y, 400.0, 180.0, eq=1000.0) for y in range(2021, 2024)]))
        r = iv.justified_pb_value(snap, "BANK", 100.0, 1.0, VCFG)
        coe = 0.07 + 0.07
        self.assertAlmostEqual(r["assumptions"]["justified_pb"], (0.18 - 0.04) / (coe - 0.04))
        self.assertAlmostEqual(r["base"], (0.18 - 0.04) / (coe - 0.04) * 10.0)

    def test_dividend_requires_consistency(self):
        snap = build_snapshot(pd.DataFrame([fy_row(2021, 400, 40, div=20), fy_row(2022, 400, 40, div=None),
                                            fy_row(2023, 400, 40, div=20)]))
        self.assertFalse(iv.dividend_value(snap, 100.0, 1.0, VCFG)["valid"])

    def test_combine_insufficient_and_renormalized(self):
        m = [iv._res("relative", True, low=80, base=100, high=120), iv._res("dcf", reason="x"),
             iv._res("historical", reason="y")]
        out = iv.combine(m, {"relative": 0.3, "dcf": 0.4, "historical": 0.3}, 0.4)
        self.assertEqual(out["status"], "INSUFFICIENT_DATA")
        m[1] = iv._res("dcf", True, low=100, base=140, high=180)
        out = iv.combine(m, {"relative": 0.3, "dcf": 0.4, "historical": 0.3}, 0.4)
        self.assertEqual(out["status"], "OK")
        self.assertAlmostEqual(out["base"], (0.3 * 100 + 0.4 * 140) / 0.7)

    def test_margin_of_safety_and_status(self):
        self.assertAlmostEqual(margin_of_safety(1000, 600), 0.4)
        self.assertIsNone(margin_of_safety(None, 600))
        th = {"deep_value": 0.4, "undervalued": 0.2, "overvalued": -0.15}
        self.assertEqual(classify(0.45, 0.1, "HIGH", th), "DEEP_VALUE")
        self.assertEqual(classify(0.45, 0.1, "LOW", th), "UNDERVALUED")   # keyakinan rendah tidak boleh DEEP_VALUE
        self.assertEqual(classify(0.45, -0.1, "HIGH", th), "UNDERVALUED")
        self.assertEqual(classify(-0.3, -0.6, "HIGH", th), "OVERVALUED")
        self.assertEqual(classify(None, None, None, th), "INSUFFICIENT_DATA")

    def test_scores_none_when_missing(self):
        self.assertIsNone(value_score(None, None, None))
        self.assertGreater(value_score(0.3, 0.1, "HIGH"), value_score(0.3, 0.1, "LOW"))
        q, _ = quality_score({"roe": 0.2}, "GENERAL")
        self.assertIsNone(q)


class ValueTrapTest(unittest.TestCase):
    def test_deteriorating_business_high_risk(self):
        hist = [fy_row(2020, 500, 50, debt=300), fy_row(2021, 450, 30, debt=500), fy_row(2022, 400, 10, debt=900),
                fy_row(2023, 350, -20, debt=1500)]
        snap = build_snapshot(pd.DataFrame(hist))
        m, _ = fundamental_ratios(snap, "GENERAL")
        t = detect(snap, m, "GENERAL", as_of="2024-05-01")
        self.assertEqual(t["risk"], "HIGH")
        self.assertTrue(any("pendapatan turun" in r for r in t["reasons"]))

    def test_unknown_when_not_enough_checks(self):
        t = detect(None, {}, "GENERAL")
        self.assertEqual(t["risk"], "UNKNOWN")


def _db(url):
    db = Database.from_url(url)
    if db.dialect == "postgres":
        db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" for t in reversed(TABLE_NAMES)])
    migrate(db)
    return db


class PointInTimeTest(unittest.TestCase):
    def test_versioning_and_as_of(self):
        for url in db_urls():
            with self.subTest(db=url.split(":")[0]):
                cfg = make_cfg()
                repo = Repository(_db(url))
                repo.upsert_stocks(pd.DataFrame({"ticker": ["TEST"], "is_active": [True]}))
                st = Statement("TEST", pd.Timestamp("2023-12-31"), "FY", "USD", items(revenue=100.0, net_income=10.0,
                               total_equity=50.0), "csv_fundamentals", "2024-01-01", publication_date=pd.Timestamp("2024-03-20"))
                self.assertEqual(store_statements(repo, cfg, [st], pd.Timestamp("2024-06-01"), True), 1)
                self.assertEqual(store_statements(repo, cfg, [st], pd.Timestamp("2024-06-02"), True), 0)  # idempoten
                self.assertTrue(load_statements(repo, "2024-03-19").empty)        # belum dipublikasikan
                self.assertEqual(len(load_statements(repo, "2024-03-20")), 1)
                restated = Statement("TEST", pd.Timestamp("2023-12-31"), "FY", "USD", items(revenue=100.0, net_income=8.0,
                                     total_equity=50.0), "csv_fundamentals", "2024-09-01",
                                     publication_date=pd.Timestamp("2024-03-20"))
                self.assertEqual(store_statements(repo, cfg, [restated], pd.Timestamp("2024-09-01"), True), 1)
                before = load_statements(repo, "2024-08-31")
                after = load_statements(repo, "2024-09-01")
                self.assertEqual(before["items"].iloc[0]["net_income"], 10.0)  # angka lama sebelum restatement diketahui
                self.assertEqual(after["items"].iloc[0]["net_income"], 8.0)
                self.assertEqual(int(repo.db.scalar("SELECT COUNT(*) FROM financial_statements")), 2)  # histori tidak hilang

    def test_snapshot_source_without_publication_date(self):
        repo = Repository(_db("sqlite:///:memory:"))
        cfg = make_cfg()
        repo.upsert_stocks(pd.DataFrame({"ticker": ["TEST"], "is_active": [True]}))
        st = Statement("TEST", pd.Timestamp("2021-12-31"), "FY", "USD", items(revenue=1.0, net_income=1.0, total_equity=1.0),
                       "yahoo_fundamentals", "2024-06-01")
        store_statements(repo, cfg, [st], pd.Timestamp("2024-06-01"), False)
        self.assertTrue(load_statements(repo, "2023-01-01").empty)              # snapshot tidak dipakai mundur
        self.assertEqual(len(load_statements(repo, "2023-01-01", use_estimated=True)), 1)  # hanya mode riset berlabel


class RunValuationTest(unittest.TestCase):
    def _seed(self, url, known_offset_days=90):
        cfg = make_cfg()
        repo = Repository(_db(url))
        tickers = [f"PE{i:02d}" for i in range(7)]
        repo.upsert_stocks(pd.DataFrame({"ticker": tickers, "is_active": True, "sector": "Consumer Non-Cyclicals",
                                         "subsector": "Food & Beverage"}))
        ids = repo.stock_ids()
        dates = pd.bdate_range("2019-01-01", "2024-06-28")
        rows = []
        for i, t in enumerate(tickers):
            px = 1000.0 + 50 * i
            rows.append(pd.DataFrame({"stock_id": ids[t], "date": dates, "open": px, "high": px, "low": px, "close": px,
                                      "volume": 1e6, "value": px * 1e6, "is_adjusted": True, "source": "fixture"}))
        repo.upsert_prices(pd.concat(rows))
        sts = []
        for i, t in enumerate(tickers):
            for k, y in enumerate(range(2018, 2024)):
                ni = (80.0 + 5 * i) * 1.05 ** k
                it = fy_row(y, ni * 10, ni, eq=ni * 6, div=ni * 0.4)["items"]
                it["shares_outstanding"] = 1.0
                sts.append(Statement(t, pd.Timestamp(f"{y}-12-31"), "FY", "USD", it, "csv_fundamentals", "2024-01-01",
                                     publication_date=pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=known_offset_days)))
        store_statements(repo, cfg, sts, pd.Timestamp("2024-07-01"), True)
        return cfg, repo

    def test_end_to_end_and_no_future_fundamentals(self):
        for url in db_urls():
            with self.subTest(db=url.split(":")[0]):
                cfg, repo = self._seed(url)
                df = run_valuation(cfg, repo, "2024-06-28")
                self.assertEqual(len(df), 7)
                ok = df[df["valuation_status"] != "INSUFFICIENT_DATA"]
                self.assertGreater(len(ok), 0)
                self.assertTrue((ok["fair_value_low"] <= ok["fair_value_base"]).all())
                self.assertTrue((ok["fair_value_base"] <= ok["fair_value_high"]).all())
                m = json.loads(ok["methods"].iloc[0])
                self.assertIn("relative", [x["method"] for x in m["methods"] if x["valid"]])
                # FY2023 dipublikasikan 2024-03-30 → valuasi 2024-03-01 hanya boleh memakai FY2022
                early = run_valuation(cfg, repo, "2024-03-01")
                self.assertTrue((pd.to_datetime(early["fundamentals_period_end"].dropna()) <= "2022-12-31").all())
                self.assertEqual(save_valuations(repo, df), 7)
                self.assertEqual(save_valuations(repo, df), 7)              # UPSERT idempoten
                self.assertEqual(int(repo.db.scalar("SELECT COUNT(*) FROM valuation_results")), 7)

    def test_no_fundamentals_is_insufficient_not_zero(self):
        cfg, repo = self._seed("sqlite:///:memory:")
        df = run_valuation(cfg, repo, "2018-06-01")
        self.assertTrue((df["valuation_status"] == "INSUFFICIENT_DATA").all())
        self.assertTrue(df["value_score"].isna().all() if "value_score" in df else True)
        self.assertTrue(np.isnan(pd.to_numeric(df.get("fair_value_base", pd.Series([np.nan])), errors="coerce")).all())


if __name__ == "__main__":
    unittest.main()
