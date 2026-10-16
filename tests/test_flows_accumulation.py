"""Flow (FINRA short volume; slot foreign flow) & akumulasi/distribusi: rumus indikator (dihitung manual), satuan, cakupan data, status, leakage.

Seluruh deret harga/flow di sini adalah FIXTURE UJI deterministik — bukan data pasar.
"""
import unittest
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from app.accumulation.accumulation_indicators import compute_indicators
from app.accumulation.accumulation_report import run_accumulation, run_flow_analysis, save_accumulation
from app.accumulation.accumulation_scorer import score
from app.accumulation.volume_analysis import rvol, up_down_volume_ratio
from app.database.db import Database
from app.database.repository import Repository
from app.database.schema import TABLE_NAMES, migrate
from app.features import indicators as ind
from app.flows.flow_features import build_flow_features
from app.flows.flow_normalizer import normalize
from app.flows.flow_scorer import score_flows
from app.flows.foreign_flow_provider import FINRAShortVolumeProvider, load_flows, store_flows, update_foreign_flow
from tests.helpers import db_urls, make_cfg


class IndicatorMathTest(unittest.TestCase):
    def test_clv_cmf_adl(self):
        h = pd.Series([10.0, 12.0, 11.0])
        l_ = pd.Series([8.0, 10.0, 11.0])          # hari ke-3 tanpa range → CLV 0
        c = pd.Series([9.5, 10.5, 11.0])
        v = pd.Series([100.0, 200.0, 300.0])
        clv = ind.clv(h, l_, c)
        self.assertEqual(clv.round(6).tolist(), [0.5, -0.5, 0.0])
        self.assertEqual(ind.adl(h, l_, c, v).tolist(), [50.0, -50.0, -50.0])
        cmf = ind.cmf(h, l_, c, v, 2)
        self.assertTrue(np.isnan(cmf.iloc[0]))
        self.assertAlmostEqual(cmf.iloc[1], (50 - 100) / 300)
        self.assertAlmostEqual(cmf.iloc[2], (-100 + 0) / 500)

    def test_mfi(self):
        h = pd.Series([10.0, 11.0, 10.5, 12.0])
        l_ = pd.Series([9.0, 10.0, 9.5, 11.0])
        c = pd.Series([9.5, 10.5, 10.0, 11.5])
        v = pd.Series([100.0, 100.0, 100.0, 100.0])
        tp = (h + l_ + c) / 3                     # 9.5, 10.5, 10.0, 11.5
        m = ind.mfi(h, l_, c, v, 3)
        pos = tp[1] * 100 + tp[3] * 100
        neg = tp[2] * 100
        self.assertAlmostEqual(m.iloc[3], 100 - 100 / (1 + pos / neg))
        self.assertEqual(ind.mfi(pd.Series([1.0, 2, 3, 4]), pd.Series([1.0, 2, 3, 4]), pd.Series([1.0, 2, 3, 4]),
                                 pd.Series([1.0] * 4), 3).iloc[3], 100.0)   # tidak ada aliran negatif

    def test_rvol_excludes_today_and_udvr(self):
        v = pd.Series([100.0, 100.0, 100.0, 400.0])
        self.assertAlmostEqual(rvol(v, 3).iloc[3], 4.0)
        c = pd.Series([1.0, 2.0, 1.5, 2.5])
        self.assertAlmostEqual(up_down_volume_ratio(c, pd.Series([0.0, 10.0, 5.0, 20.0]), 3).iloc[3], 30 / 5)

    def test_vwap_uses_traded_value(self):
        h = l_ = c = pd.Series([100.0, 100.0])
        v = pd.Series([10.0, 30.0])
        value = pd.Series([1000.0, 3300.0])        # harga transaksi riil berbeda dari close
        self.assertAlmostEqual(ind.rolling_vwap(h, l_, c, v, 2, value).iloc[1], 4300 / 40)


class FlowNormalizerTest(unittest.TestCase):
    def test_lots_values_and_unit_check(self):
        raw = pd.DataFrame({"Kode Saham": ["AAAA", "BBBB"], "date": ["2024-06-28"] * 2, "Foreign Buy": [10, 5000],
                            "Foreign Sell": [4, 10], "Volume": [100, 1000], "units": ["lot", "shares"]})
        n = normalize(raw, "t")
        a = n[n["ticker"] == "AAAA"].iloc[0]
        self.assertEqual(a["foreign_buy_shares"], 1000)          # 10 lot × 100
        self.assertEqual(a["net_foreign_shares"], 600)
        self.assertEqual(a["value_type"], "SHARES_ONLY")         # nilai tidak dikarang
        self.assertTrue(np.isnan(a["foreign_buy_value"]))
        self.assertEqual(n[n["ticker"] == "BBBB"].iloc[0]["quality_status"], "CHECK")   # beli asing > volume

    def test_finra_short_volume_file(self):
        """Format resmi FINRA: header + baris pipe + trailer jumlah record; BRK.B → BRK-B; buy = total − short."""
        txt = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
               "20240628|AAPL|4000|10|10000|B,Q,N\n"
               "20240628|BRK.B|600|0|1000|Q,N\n"
               "20240628|ZERO|0|0|0|N\n"
               "3\n")
        raw = FINRAShortVolumeProvider.parse(txt, pd.Timestamp("2024-06-28"))
        self.assertEqual(len(raw), 3)
        n = normalize(raw.assign(ticker=raw["ticker"].str.replace(".", "-", regex=False)), "finra_short_volume", lot_size=1)
        a = n.set_index("ticker")
        self.assertEqual(a.loc["AAPL", "foreign_sell_shares"], 4000)          # short volume
        self.assertEqual(a.loc["AAPL", "foreign_buy_shares"], 6000)           # non-short volume
        self.assertEqual(a.loc["AAPL", "net_foreign_shares"], 2000)           # total − 2 × short
        self.assertEqual(a.loc["BRK-B", "total_volume_shares"], 1000)
        self.assertEqual(a.loc["ZERO", "quality_status"], "NO_TRADE")
        empty = FINRAShortVolumeProvider.parse("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n0\n")
        self.assertTrue(empty.empty)

    def test_finra_fetch_uses_trading_days_only(self):
        """Hanya hari bursa NYSE yang diunduh; file yang belum terbit (None) dilewati tanpa error."""
        got = []

        class P(FINRAShortVolumeProvider):
            def _download(self, day):
                got.append(day)
                if day == pd.Timestamp("2024-07-05"):
                    return None
                return (f"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                        f"{day:%Y%m%d}|AAPL|10|0|40|Q\n1\n")
        p = P({"name": "finra", "backfill_trading_days": 50}, make_cfg())
        df = p.fetch(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-08"))
        self.assertNotIn(pd.Timestamp("2024-07-04"), got)                   # Independence Day (NYSE tutup)
        self.assertEqual(sorted(df["date"].unique()), [pd.Timestamp(d) for d in
                                                       ("2024-07-01", "2024-07-02", "2024-07-03", "2024-07-08")])
        self.assertTrue((df["net_foreign_shares"] == 20).all())


def _prices(n=140, tickers=("AAAA",), trend=0.0, close_pos=0.8, vol_up=1.0, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for t in tickers:
        c = 1000 * np.cumprod(1 + trend + rng.normal(0, 0.01, n))
        rng_ = c * 0.02
        low = c - rng_ * close_pos
        high = low + rng_
        vol = 1e6 * np.where(np.diff(np.r_[c[0], c]) > 0, vol_up, 1.0)
        rows.append(pd.DataFrame({"ticker": t, "date": dates, "open": c, "high": high, "low": low, "close": c,
                                  "volume": vol, "value": c * vol}))
    return pd.concat(rows, ignore_index=True)


class FlowFeatureTest(unittest.TestCase):
    def test_missing_days_reduce_coverage_not_zero(self):
        p = _prices(30)
        dates = p["date"].tolist()
        f = pd.DataFrame({"ticker": "AAAA", "date": dates[-10:], "foreign_buy_shares": 3e5, "foreign_sell_shares": 1e5,
                          "net_foreign_shares": 2e5, "net_foreign_value": np.nan, "market_segment": "TOTAL",
                          "quality_status": "OK"})
        feat = build_flow_features(p, f)
        last = feat.iloc[-1]
        self.assertAlmostEqual(last["flow_coverage_20"], 0.5)
        self.assertAlmostEqual(last["flow_net_shares_20"], 2e6)
        self.assertAlmostEqual(last["flow_net_ratio_20"], 2e6 / 1e7)       # pembagi hanya volume hari ber-data
        self.assertEqual(last["flow_value_type_20"], "ESTIMATED_VALUE")
        sc = score_flows(feat, 0.8).iloc[-1]
        self.assertEqual(sc["foreign_flow_status"], "INSUFFICIENT_DATA")    # cakupan 50% → tidak dinilai
        self.assertTrue(np.isnan(sc["foreign_flow_score"]))
        self.assertTrue(np.isnan(feat.iloc[5]["flow_net_shares_20"]))      # sebelum ada data: NaN, bukan 0

    def test_strong_buying_and_no_lookahead(self):
        p = _prices(80)
        dates = p["date"].tolist()
        f = pd.DataFrame({"ticker": "AAAA", "date": dates, "foreign_buy_shares": 4e5, "foreign_sell_shares": 1e5,
                          "net_foreign_shares": 3e5, "net_foreign_value": np.nan, "market_segment": "TOTAL",
                          "quality_status": "OK"})
        feat = build_flow_features(p, f)
        sc = score_flows(feat, 0.8)
        self.assertEqual(sc.iloc[-1]["foreign_flow_status"], "STRONG_NET_BUYING")
        # menambah data masa depan tidak mengubah fitur hari sebelumnya
        f2 = f.copy()
        f2.loc[f2.index[-5:], "net_foreign_shares"] = -9e6
        feat2 = build_flow_features(p, f2)
        cut = len(p) - 6
        pd.testing.assert_series_equal(feat["flow_net_ratio_20"].iloc[:cut], feat2["flow_net_ratio_20"].iloc[:cut])

    def test_baseline_measures_deviation_from_own_history(self):
        """FINRA: rasio short struktural ±50% tidak boleh terbaca sebagai 'net beli'. Yang dinilai simpangannya."""
        p = _prices(200)
        dates = p["date"].tolist()
        short = np.r_[np.full(170, 5.0e5), np.full(30, 3.0e5)]                # short turun di 30 hari terakhir
        f = pd.DataFrame({"ticker": "AAAA", "date": dates, "foreign_sell_shares": short,
                          "foreign_buy_shares": 1e6 - short, "net_foreign_shares": 1e6 - 2 * short,
                          "total_volume_shares": 1e6, "net_foreign_value": np.nan, "market_segment": "TOTAL",
                          "quality_status": "OK"})
        feat = build_flow_features(p, f, baseline_days=120)
        mid = feat.iloc[160]                                                   # rasio stabil → simpangan 0
        self.assertAlmostEqual(mid["flow_net_ratio_20"], 0.0)
        last = feat.iloc[-1]
        self.assertGreater(last["flow_net_ratio_20"], 0.25)                    # 0,4 sekarang vs baseline ±0,13
        self.assertEqual(score_flows(feat, 0.8).iloc[-1]["foreign_flow_status"], "STRONG_NET_BUYING")
        # baseline hari D tidak memakai data D atau sesudahnya
        f2 = f.copy()
        f2.loc[f2.index[-1], ["foreign_sell_shares", "net_foreign_shares"]] = [1e6, -1e6]
        feat2 = build_flow_features(p, f2, baseline_days=120)
        self.assertAlmostEqual(feat.iloc[-2]["flow_net_ratio_20"], feat2.iloc[-2]["flow_net_ratio_20"])


class AccumulationScoreTest(unittest.TestCase):
    def test_strong_requires_foreign_flow(self):
        p = _prices(140, trend=0.004, close_pos=0.9, vol_up=2.5)
        indic = compute_indicators(p)
        last = indic.tail(1)
        no_flow = score(last, None)
        self.assertNotEqual(no_flow.iloc[0]["accumulation_status"], "STRONG_ACCUMULATION_SIGNAL")
        self.assertIn(no_flow.iloc[0]["accumulation_status"], ("MODERATE_ACCUMULATION_SIGNAL",))
        self.assertTrue(no_flow.iloc[0]["_strong_capped"])
        flow = last[["ticker", "date"]].assign(foreign_flow_score=85.0, foreign_flow_status="STRONG_NET_BUYING")
        with_flow = score(last, flow)
        self.assertEqual(with_flow.iloc[0]["accumulation_status"], "STRONG_ACCUMULATION_SIGNAL")

    def test_distribution_pattern(self):
        p = _prices(140, trend=-0.004, close_pos=0.1, vol_up=0.4)
        r = score(compute_indicators(p).tail(1), None).iloc[0]
        self.assertIn(r["accumulation_status"], ("MODERATE_DISTRIBUTION_SIGNAL",))
        self.assertEqual(r["accumulation_stage"], "DISTRIBUTION_WARNING")

    def test_indicators_have_no_lookahead(self):
        p = _prices(160, ("AAAA", "BBBB"), trend=0.002, vol_up=1.5)
        full = compute_indicators(p)
        cut_date = sorted(p["date"].unique())[110]
        part = compute_indicators(p[p["date"] <= cut_date])
        a = full[full["date"] == cut_date].set_index("ticker").sort_index()
        b = part[part["date"] == cut_date].set_index("ticker").sort_index()
        cols = [c for c in a.columns if c.startswith("acc_")]
        pd.testing.assert_frame_equal(a[cols], b[cols])             # data setelah tanggal tidak memengaruhi nilai

    def test_short_history_insufficient(self):
        r = score(compute_indicators(_prices(40)).tail(1), None).iloc[0]
        self.assertEqual(r["accumulation_status"], "INSUFFICIENT_DATA")
        self.assertTrue(np.isnan(r["accumulation_score"]))


def _db(url):
    db = Database.from_url(url)
    if db.dialect == "postgres":
        db.executescript([f"DROP TABLE IF EXISTS {t} CASCADE" for t in reversed(TABLE_NAMES)])
    migrate(db)
    return db


class FlowStorageTest(unittest.TestCase):
    def test_store_idempotent_and_pipeline(self):
        for url in db_urls():
            with self.subTest(db=url.split(":")[0]):
                repo = Repository(_db(url))
                cfg = make_cfg()
                cfg["foreign_flow"]["baseline_days"] = None      # sumber beli/jual sungguhan (tanpa bias struktural)
                repo.upsert_stocks(pd.DataFrame({"ticker": ["AAAA", "BBBB"], "is_active": True}))
                p = _prices(140, ("AAAA", "BBBB"), trend=0.003, vol_up=2.0)
                dates = sorted(p["date"].unique())
                f = normalize(pd.DataFrame({"ticker": "AAAA", "date": dates, "foreign_buy_shares": 4e5,
                                            "foreign_sell_shares": 1e5, "total_volume_shares": 1e6}), "fixture")
                self.assertEqual(store_flows(repo, f), len(dates))
                store_flows(repo, f)
                self.assertEqual(int(repo.db.scalar("SELECT COUNT(*) FROM foreign_flow_history")), len(dates))
                self.assertEqual(len(load_flows(repo, end=dates[9])), 10)
                fl = run_flow_analysis(cfg, repo, p, dates[-1])
                last = fl[fl["date"] == dates[-1]].set_index("ticker")
                self.assertEqual(last.loc["BBBB", "foreign_flow_status"], "FOREIGN_FLOW_UNAVAILABLE")
                self.assertIn(last.loc["AAAA", "foreign_flow_status"], ("STRONG_NET_BUYING", "NET_BUYING"))
                acc = run_accumulation(cfg, repo, p, dates[-1], fl)
                self.assertEqual(len(acc), 2)
                b = acc.set_index("ticker").loc["BBBB"]
                self.assertEqual(b["foreign_flow_status"], "FOREIGN_FLOW_UNAVAILABLE")
                n = save_accumulation(cfg, repo, acc, "run1")
                self.assertEqual(save_accumulation(cfg, repo, acc, "run1"), n)
                self.assertEqual(int(repo.db.scalar("SELECT COUNT(*) FROM accumulation_signals")), n)

    def test_update_without_sources_is_unavailable(self):
        repo = Repository(_db("sqlite:///:memory:"))
        cfg = make_cfg()
        cfg["foreign_flow"]["providers"] = [{"name": "x", "type": "csv", "enabled": True, "dir": mkdtemp()},
                                            {"name": "y", "type": "http", "enabled": False}]
        st = update_foreign_flow(cfg, repo, "2024-06-28")
        self.assertEqual(st["status"], "FOREIGN_FLOW_UNAVAILABLE")
        self.assertEqual(st["providers"]["x"]["status"], "NOT_CONFIGURED")
        self.assertEqual(repo.db.scalar("SELECT status FROM data_sources WHERE name = 'y'"), "DISABLED")


if __name__ == "__main__":
    unittest.main()
