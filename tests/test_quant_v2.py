"""Perbaikan quant v2 (audit 16 Okt 2026): label residual cross-sectional, fitur alpha, earnings PIT, gerbang biaya,
statistik signifikansi, dan sumber data yang tidak lagi gagal diam-diam. Semua data di sini FIXTURE deterministik."""
import unittest

import numpy as np
import pandas as pd

from app.backtest.stats import alpha_beta, deflated_sharpe, probabilistic_sharpe, sharpe_ci, significance_report
from app.features.alpha import add_cross_sectional_ranks, add_market_relative
from app.features.earnings import earnings_features, reaction_date
from app.models.evaluation import rank_ic
from app.models.labels import add_class_label
from app.valuation.sec_edgar import earnings_from_submissions
from tests.helpers import make_cfg


def _panel(n_days=320, n_stocks=30, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    m = rng.normal(0.0004, 0.01, n_days)
    idx = 4000 * np.cumprod(1 + m)
    rows = []
    for i in range(n_stocks):
        beta = 0.5 + i / n_stocks * 1.5
        r = beta * m + rng.normal(0, 0.012, n_days)
        c = 50 * np.cumprod(1 + r)
        rows.append(pd.DataFrame({"ticker": f"T{i:02d}", "date": dates, "close": c, "high": c * 1.01, "low": c * 0.99,
                                  "idx_close": idx, "true_beta": beta}))
    df = pd.concat(rows, ignore_index=True)
    g = df.groupby("ticker")["close"]
    for n in (5, 20):
        df[f"ret{n}"] = g.pct_change(n)
    ix = df[["date", "idx_close"]].drop_duplicates("date").set_index("date")["idx_close"]
    for n in (5, 20):
        df[f"idx_ret{n}"] = df["date"].map(ix / ix.shift(n) - 1)
    df["vol20"] = g.transform(lambda s: s.pct_change().rolling(20).std())
    df["is_tradeable"] = True
    return df


class TestMarketRelative(unittest.TestCase):
    def test_beta_recovered_and_no_lookahead(self):
        df = add_market_relative(_panel())
        last = df[df["date"] == df["date"].max()]
        err = (last["f_beta60"] - last["true_beta"]).abs()
        self.assertLess(err.median(), 0.25)                       # beta 60H mendekati beta sebenarnya
        cut = df["date"].unique()[200]
        a = add_market_relative(_panel()[lambda d: d["date"] <= cut])
        b = df[df["date"] <= cut]
        pd.testing.assert_series_equal(a.set_index(["ticker", "date"])["f_beta60"].sort_index(),
                                       b.set_index(["ticker", "date"])["f_beta60"].sort_index())

    def test_cross_sectional_rank_uses_same_day_only(self):
        p = _panel()
        p["f_ret5"] = p["ret5"]
        p.loc[p["ticker"] == "T00", "is_tradeable"] = False
        out = add_cross_sectional_ranks(p, ["f_ret5"])
        d = out[out["date"] == out["date"].max()]
        self.assertTrue(d.loc[d["ticker"] == "T00", "f_xs_ret5"].isna().all())   # tidak bisa ditransaksikan → NaN
        self.assertAlmostEqual(d["f_xs_ret5"].max(), 1.0)


class TestResidualLabel(unittest.TestCase):
    def test_label_is_market_and_vol_neutral(self):
        cfg = make_cfg()
        cfg["labels"]["mode"] = "xs_resid"
        df = add_market_relative(_panel(seed=3))
        h = cfg["labels"]["SWING_HORIZON"]
        df[f"y_ret_{h}d"] = df.groupby("ticker")["close"].transform(lambda s: s.shift(-h) / s - 1)
        out = add_class_label(df, cfg)
        lab = out.dropna(subset=["y_class"])
        share = lab.groupby("date")["y_class"].apply(lambda s: (s == 2).mean())
        self.assertAlmostEqual(share.mean(), 1 / 3, delta=0.04)   # sepertiga teratas per tanggal
        # label lama (±3% mentah) didominasi saham beta tinggi di pasar naik; label baru tidak
        hi_beta = lab["true_beta"] > 1.5
        self.assertLess(abs(lab.loc[hi_beta, "y_class"].eq(2).mean() - lab.loc[~hi_beta, "y_class"].eq(2).mean()), 0.12)

    def test_fixed_mode_still_available(self):
        cfg = make_cfg()
        cfg["labels"]["mode"] = "fixed"
        df = pd.DataFrame({"date": pd.Timestamp("2024-01-02"), f"y_ret_{cfg['labels']['SWING_HORIZON']}d": [0.05, 0.0, -0.05]})
        self.assertEqual(add_class_label(df, cfg)["y_class"].tolist(), [2.0, 1.0, 0.0])


class TestEarnings(unittest.TestCase):
    def test_reaction_day_and_pit(self):
        rd = reaction_date(pd.Series(["2024-02-01T16:30:00.000Z", "2024-05-02T08:00:00.000Z", "2024-11-01T17:00:00.000Z"]),
                           pd.Series(["2024-02-01", "2024-05-02", "2024-11-01"]))
        self.assertEqual([str(d.date()) for d in rd], ["2024-02-02", "2024-05-02", "2024-11-04"])
        d = pd.bdate_range("2024-01-01", "2024-12-31")
        df = pd.DataFrame({"ticker": "AAA", "date": d, "close": 100.0, "idx_close": 1000.0})
        df.loc[df["date"] >= rd[0], "close"] = 108.0
        out = earnings_features(df, pd.DataFrame({"ticker": "AAA", "reaction_date": rd})).set_index("date")
        self.assertTrue(np.isnan(out.loc["2024-02-02", "f_ear3"]))           # reaksi belum lengkap
        self.assertAlmostEqual(out.loc["2024-02-05", "f_ear3"], 0.08, places=6)
        self.assertEqual(out.loc["2024-02-02", "f_days_since_earn"], 0)
        self.assertTrue(np.isnan(out.loc["2024-01-31", "f_days_since_earn"]))  # sebelum event pertama: tidak ada data

    def test_submissions_parsing(self):
        block = {"accessionNumber": ["a1", "a2", "a3", "a4"], "filingDate": ["2024-02-01", "2024-02-15", "2024-05-02", "2024-05-03"],
                 "acceptanceDateTime": ["2024-02-01T16:30:00.000Z", "2024-02-15T09:00:00.000Z", "2024-05-02T16:05:00.000Z",
                                        "2024-05-03T10:00:00.000Z"],
                 "form": ["8-K", "8-K", "8-K", "8-K/A"], "items": ["2.02,9.01", "5.02", "2.02", "2.02"]}
        ev = earnings_from_submissions([block])
        self.assertEqual(ev["accession"].tolist(), ["a1", "a3"])             # Item 2.02 saja, tanpa amandemen
        self.assertEqual([str(x.date()) for x in ev["reaction_date"]], ["2024-02-02", "2024-05-03"])


class TestEarningsWatch(unittest.TestCase):
    def test_watchlist_targets_expected_release_window(self):
        from app.valuation.fundamental_provider import earnings_watchlist
        q = pd.to_datetime(["2025-01-30", "2025-04-30", "2025-07-30", "2025-10-29"])
        ev = pd.concat([pd.DataFrame({"ticker": "DUE", "reaction_date": q}),
                        pd.DataFrame({"ticker": "LATER", "reaction_date": q - pd.Timedelta(days=45)}),
                        pd.DataFrame({"ticker": "FRESH", "reaction_date": q.append(pd.DatetimeIndex(["2026-01-25"]))})])
        # perkiraan rilis berikutnya DUE ≈ 2026-01-28 (jeda median ±91 hari)
        self.assertEqual(earnings_watchlist(ev, "2026-01-30"), ["DUE"])
        self.assertEqual(earnings_watchlist(ev, "2025-12-01"), [])


class TestRetrainSkipIsCheap(unittest.TestCase):
    def test_if_due_skips_before_building_dataset(self):
        from types import SimpleNamespace
        from unittest import mock
        from app import FEATURE_VERSION
        import app.pipeline.jobs as jobs

        class Models:
            def active_version(self):
                return {"version": "model_v003", "feature_version": FEATURE_VERSION,
                        "promoted_at": pd.Timestamp.now(tz="UTC").isoformat(), "created_at": None}
        ctx = SimpleNamespace(cfg=make_cfg(), models=Models())
        with mock.patch.object(jobs, "build_dataset", side_effect=AssertionError("dataset tidak boleh dibangun")):
            out = jobs.job_train(ctx, force=False)
        self.assertEqual(out["decision"], "SKIPPED")
        self.assertIn("force=true", out["reason"])


class TestFilters(unittest.TestCase):
    def _run(self, earn_days, er):
        from app.strategy.filters import apply_filters
        cfg = make_cfg()
        n = len(earn_days)
        idx = pd.RangeIndex(n)
        df = pd.DataFrame({"market_regime": "BULL", "is_liquid": True, "is_suspended": False, "days_listed": 999,
                           "close": 100.0, "ca_recent": False, "atr_pct": 0.02, "earn_soon_days": earn_days}, index=idx)
        setups = pd.DataFrame({"setup_valid": True, "is_low_quality_breakout": False, "setup_forming": False}, index=idx)
        pred = pd.DataFrame({"probability_bullish": 0.6, "probability_bearish": 0.1, "prob_dispersion": 0.0,
                             "base_rate_bullish": 1 / 3, f"expected_return_{cfg['labels']['SWING_HORIZON']}d": er}, index=idx)
        stops = pd.DataFrame({"stop_too_wide": False}, index=idx)
        targets = pd.DataFrame({"risk_reward": 2.0}, index=idx)
        sizing = pd.DataFrame({"position_lots": 10}, index=idx)
        scores = pd.DataFrame({"final_score": 80.0}, index=idx)
        return apply_filters(df, setups, pred, stops, targets, sizing, scores, cfg)

    def test_earnings_blackout_and_missing_data(self):
        out = self._run([5.0, 40.0, np.nan], 0.02)
        self.assertIn("earnings_within_holding", out.loc[0, "reject_reasons"])
        self.assertNotIn("earnings_within_holding", out.loc[1, "reject_reasons"])
        self.assertNotIn("earnings_within_holding", out.loc[2, "reject_reasons"])    # tanpa data → tidak memblokir

    def test_expected_return_must_beat_costs(self):
        out = self._run([np.nan, np.nan], [0.001, 0.02])                             # biaya pulang-pergi ≈ 0,31%
        self.assertIn("expected_return_below_cost", out.loc[0, "reject_reasons"])
        self.assertNotIn("expected_return_below_cost", out.loc[1, "reject_reasons"])


class TestSignificance(unittest.TestCase):
    def test_alpha_beta_and_deflation(self):
        rng = np.random.default_rng(0)
        d = pd.bdate_range("2023-01-02", periods=500)
        m = pd.Series(rng.normal(0.0004, 0.01, 500), index=d)
        s = 0.5 * m + pd.Series(rng.normal(0.0008, 0.004, 500), index=d)
        ab = alpha_beta(s, m)
        self.assertAlmostEqual(ab["beta"], 0.5, delta=0.05)
        self.assertGreater(ab["alpha_annual"], 0.1)
        self.assertLess(deflated_sharpe(s, 200), probabilistic_sharpe(s))          # banyak percobaan → bukti dipotong
        ci = sharpe_ci(s)
        self.assertLess(ci["sharpe_ci_low"], ci["sharpe_ci_high"])
        eq = pd.DataFrame({"date": d, "equity": 10000 * np.cumprod(1 + s.to_numpy())})
        rep = significance_report(eq, pd.Series(4000 * np.cumprod(1 + m.to_numpy()), index=d), n_trials=10)
        self.assertIn("verdict", rep)

    def test_rank_ic(self):
        rng = np.random.default_rng(1)
        rows = []
        for d in pd.bdate_range("2024-01-01", periods=60):
            x = rng.normal(size=50)
            rows.append(pd.DataFrame({"date": d, "score": x, "target": 0.3 * x + rng.normal(size=50)}))
        r = rank_ic(pd.concat(rows), "score", "target")
        self.assertGreater(r["rank_ic_mean"], 0.15)
        self.assertGreater(r["rank_ic_tstat"], 5)


class TestDataSourcesNotSilent(unittest.TestCase):
    def test_finra_block_raises(self):
        from app.flows.foreign_flow_provider import FINRAShortVolumeProvider

        class Blocked(FINRAShortVolumeProvider):
            def _download(self, day):
                self._blocked += 1
                return None
        p = Blocked({"name": "finra", "backfill_trading_days": 30}, make_cfg())
        with self.assertRaises(RuntimeError):
            p.fetch(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-31"))

    def test_wikipedia_picks_constituents_not_changes_table(self):
        # Halaman S&P 400: tabel riwayat (Added/Removed, dua kolom "Ticker") lebih panjang dari tabel konstituen.
        from app.data.us_universe import parse_wikipedia
        cons = "".join(f"<tr><td>T{i}</td><td>Co {i}</td><td>Industrials</td><td>Machinery</td></tr>" for i in range(5))
        chg = "".join(f"<tr><td>2024-01-{i % 28 + 1:02d}</td><td>A{i}</td><td>Add {i}</td><td>R{i}</td><td>Rem {i}</td>"
                      f"<td>x</td></tr>" for i in range(8))
        html = (f"<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>{cons}"
                "</table><table><tr><th rowspan=2>Date</th><th colspan=2>Added</th><th colspan=2>Removed</th>"
                "<th rowspan=2>Reason</th></tr><tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th>"
                f"</tr>{chg}</table>")
        out = parse_wikipedia(html, "S&P 400", 3)
        self.assertEqual(out["ticker"].tolist(), [f"T{i}" for i in range(5)])
        self.assertEqual(set(out["sector"]), {"Industrials"})
        with self.assertRaises(ValueError):                                  # terlalu sedikit → error, bukan diam
            parse_wikipedia(html, "S&P 400", 6)

    def test_wikipedia_user_agent_has_contact(self):
        import os
        from app.data.us_universe import user_agent
        os.environ["SEC_USER_AGENT"] = "Budi Santoso budi@example.com"
        try:
            self.assertIn("budi@example.com", user_agent())
        finally:
            os.environ.pop("SEC_USER_AGENT", None)


if __name__ == "__main__":
    unittest.main()
