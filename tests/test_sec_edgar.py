"""SEC EDGAR companyfacts → laporan point-in-time.

Fixture = companyfacts SINTETIS (struktur sama dengan data.sec.gov/api/xbrl/companyfacts), tahun fiskal berakhir
September (pola Apple): 10-Q Q1–Q3 (laba rugi kuartalan langsung, arus kas YTD), 10-K FY tanpa Q4 terpisah, lalu
10-K tahun berikutnya yang me-RESTATE angka FY lama. Bukan data emiten sungguhan.
"""
import unittest

import pandas as pd

from app.database.db import Database
from app.database.repository import Repository
from app.database.schema import migrate
from app.valuation.fundamental_provider import load_statements, store_statements
from app.valuation.sec_edgar import SECEdgarFundamentalProvider, statements_from_facts
from app.valuation.valuation_metrics import build_snapshot
from tests.helpers import make_cfg

FY_START = "2023-10-01"
Q_ENDS = ["2023-12-30", "2024-03-30", "2024-06-29", "2024-09-28"]
Q_STARTS = ["2023-10-01", "2023-12-31", "2024-03-31", "2024-06-30"]
REV_Q = [100.0, 90.0, 80.0, 110.0]
NI_Q = [20.0, 18.0, 15.0, 25.0]
OCF_Q = [30.0, 10.0, 25.0, 35.0]
FILED = {"Q1": "2024-02-02", "Q2": "2024-05-03", "Q3": "2024-08-02", "FY": "2024-11-01", "FY25": "2025-10-31"}
ACCN = {k: f"0000000000-24-{i:06d}" for i, k in enumerate(FILED)}


def _e(start, end, val, k, form, fp):
    d = {"end": end, "val": val, "accn": ACCN[k], "fy": 2024, "fp": fp, "form": form, "filed": FILED[k]}
    if start:
        d["start"] = start
    return d


def _ytd(lst, n):
    return sum(lst[:n])


def companyfacts() -> dict:
    rev, ni, ocf, assets, eq, eps = [], [], [], [], [], []
    for i, (k, fp) in enumerate((("Q1", "Q1"), ("Q2", "Q2"), ("Q3", "Q3"))):
        end = Q_ENDS[i]
        rev.append(_e(Q_STARTS[i], end, REV_Q[i], k, "10-Q", fp))                 # kuartal langsung
        ni.append(_e(Q_STARTS[i], end, NI_Q[i], k, "10-Q", fp))
        eps.append(_e(Q_STARTS[i], end, NI_Q[i] / 10, k, "10-Q", fp))
        if i:                                                                     # laba rugi YTD juga dilaporkan
            rev.append(_e(FY_START, end, _ytd(REV_Q, i + 1), k, "10-Q", fp))
            ni.append(_e(FY_START, end, _ytd(NI_Q, i + 1), k, "10-Q", fp))
        ocf.append(_e(FY_START, end, _ytd(OCF_Q, i + 1), k, "10-Q", fp))          # arus kas HANYA YTD
        assets.append(_e(None, end, 1000.0 + i, k, "10-Q", fp))
        eq.append(_e(None, end, 400.0 + i, k, "10-Q", fp))
        if i:                                                                     # komparatif tahun lalu (diabaikan)
            assets.append(_e(None, "2023-09-30", 950.0, k, "10-Q", fp))
    fy_end = Q_ENDS[3]
    rev += [_e(FY_START, fy_end, sum(REV_Q), "FY", "10-K", "FY"), _e("2022-10-01", "2023-09-30", 300.0, "FY", "10-K", "FY")]
    ni += [_e(FY_START, fy_end, sum(NI_Q), "FY", "10-K", "FY")]
    eps += [_e(FY_START, fy_end, sum(NI_Q) / 10, "FY", "10-K", "FY")]
    ocf += [_e(FY_START, fy_end, sum(OCF_Q), "FY", "10-K", "FY")]
    assets += [_e(None, fy_end, 1100.0, "FY", "10-K", "FY"), _e(None, "2023-09-30", 950.0, "FY", "10-K", "FY")]
    eq += [_e(None, fy_end, 450.0, "FY", "10-K", "FY")]
    # 10-K tahun berikutnya: me-restate revenue FY2024 menjadi 999 → TIDAK boleh mengubah angka FY2024 yang dipakai
    rev.append(_e(FY_START, fy_end, 999.0, "FY25", "10-K", "FY"))
    rev.append(_e("2024-09-29", "2025-09-27", 500.0, "FY25", "10-K", "FY"))
    assets.append(_e(None, "2025-09-27", 1200.0, "FY25", "10-K", "FY"))
    eq.append(_e(None, "2025-09-27", 470.0, "FY25", "10-K", "FY"))
    gaap = {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": rev}},
            "NetIncomeLoss": {"units": {"USD": ni}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}},
            "Assets": {"units": {"USD": assets}}, "StockholdersEquity": {"units": {"USD": eq}},
            "EarningsPerShareDiluted": {"units": {"USD/shares": eps}}}
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2024-10-18", "val": 10.0, "accn": ACCN["FY"], "fy": 2024, "fp": "FY", "form": "10-K", "filed": FILED["FY"]}]}}}
    return {"cik": 1, "entityName": "Fixture Corp", "facts": {"us-gaap": gaap, "dei": dei}}


class TestStatementsFromFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sts = statements_from_facts("FIXT", companyfacts(), max_years=10, today=pd.Timestamp("2026-01-01"))
        cls.by = {(s.period_type, str(pd.Timestamp(s.period_end).date())): s for s in sts}

    def test_periods_and_labels(self):
        keys = set(self.by)
        for pt, end in (("Q1", Q_ENDS[0]), ("Q2", Q_ENDS[1]), ("Q3", Q_ENDS[2]), ("FY", Q_ENDS[3]), ("Q4", Q_ENDS[3])):
            self.assertIn((pt, end), keys)

    def test_publication_date_is_filed_date(self):
        self.assertEqual(self.by[("Q2", Q_ENDS[1])].publication_date, pd.Timestamp(FILED["Q2"]))
        self.assertEqual(self.by[("FY", Q_ENDS[3])].publication_date, pd.Timestamp(FILED["FY"]))
        self.assertEqual(self.by[("Q4", Q_ENDS[3])].publication_date, pd.Timestamp(FILED["FY"]))

    def test_quarterly_values_discrete(self):
        for i, pt in enumerate(("Q1", "Q2", "Q3")):
            it = self.by[(pt, Q_ENDS[i])].items
            self.assertEqual(it["revenue"], REV_Q[i])
            self.assertEqual(it["operating_cash_flow"], OCF_Q[i])                 # YTD dipecah jadi kuartalan
        q4 = self.by[("Q4", Q_ENDS[3])].items
        self.assertEqual(q4["revenue"], REV_Q[3])                                  # FY − YTD 9 bulan
        self.assertEqual(q4["net_income"], NI_Q[3])
        self.assertEqual(q4["operating_cash_flow"], OCF_Q[3])

    def test_original_not_restated_and_instants(self):
        fy = self.by[("FY", Q_ENDS[3])].items
        self.assertEqual(fy["revenue"], sum(REV_Q))                                # bukan 999 dari filing 2025
        self.assertEqual(fy["total_assets"], 1100.0)
        self.assertEqual(fy["shares_outstanding"], 10.0)                           # dei: tanggal sampul
        self.assertIsNone(fy["total_debt"])                                        # tag tidak ada = None, bukan 0
        self.assertEqual(self.by[("Q1", Q_ENDS[0])].items["total_assets"], 1000.0)

    def test_ttm_snapshot_and_point_in_time(self):
        repo = Repository(Database.from_url("sqlite:///:memory:"))
        migrate(repo.db)
        repo.upsert_stocks(pd.DataFrame({"ticker": ["FIXT"], "is_active": [True]}))
        cfg = make_cfg()
        store_statements(repo, cfg, list(self.by.values()), pd.Timestamp("2026-01-01"), True)
        before_10k = load_statements(repo, "2024-10-15")
        self.assertEqual(pd.Timestamp(before_10k["period_end"].max()), pd.Timestamp(Q_ENDS[2]))  # FY belum terbit
        snap = build_snapshot(load_statements(repo, "2024-11-15"))
        self.assertEqual(snap["basis"], "FY")
        self.assertTrue(snap["quarters_discrete"])                                 # Q1..Q4 = FY → terbukti diskret
        self.assertAlmostEqual(snap["items"]["revenue"], sum(REV_Q))


class TestProviderConfig(unittest.TestCase):
    def test_requires_user_agent(self):
        cfg = make_cfg()
        p = SECEdgarFundamentalProvider({"name": "sec", "user_agent": ""}, cfg)
        self.assertEqual(p.availability()[0], "NOT_CONFIGURED")
        p = SECEdgarFundamentalProvider({"name": "sec", "user_agent": "Budi budi@example.com"}, cfg)
        self.assertEqual(p.availability()[0], "AVAILABLE")

    def test_cik_lookup_uses_internal_ticker_format(self):
        p = SECEdgarFundamentalProvider({"name": "sec", "user_agent": "Budi budi@example.com"}, make_cfg())
        p._ciks = None
        p._get_json = lambda url: {"0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"},
                                   "1": {"cik_str": 1, "ticker": "FIXT", "title": "Fixture"}}
        p.cache = p.cache.parent / "sec_test_cache"
        import shutil
        shutil.rmtree(p.cache, ignore_errors=True)
        self.assertEqual(p.cik_map()["BRK-B"], 1067983)
        calls = []
        p._get_json = lambda url: calls.append(url) or (companyfacts() if "companyfacts" in url else None)
        sts = p.get_statements("fixt")
        self.assertTrue(any("CIK0000000001.json" in u for u in calls))
        self.assertGreater(len(sts), 0)


if __name__ == "__main__":
    unittest.main()
