"""Alpha Vantage: parser, kuota 24 jam, deteksi rate-limit/premium, penyamaran key, penjadwal, dan fitur PIT.
Semua respons adalah FIXTURE berformat sama dengan API asli — tanpa internet."""
import json
import os
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from app.data.alpha_vantage import (AlphaVantageClient, AlphaVantageError, PremiumOnly, QuotaExhausted, parse_calendar,
                                    parse_earnings, parse_news, parse_series)
from tests.helpers import make_cfg, make_ctx

KEY = "TESTKEY123456789"

EARN = {"symbol": "AAA", "annualEarnings": [{"fiscalDateEnding": "2025-12-31", "reportedEPS": "9.1"}],
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-23", "reportedEPS": "2.10", "estimatedEPS": "2.00",
             "surprise": "0.10", "surprisePercentage": "5.0", "reportTime": "post-market"},
            {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-04-22", "reportedEPS": "1.90", "estimatedEPS": "1.95",
             "surprise": "-0.05", "surprisePercentage": "-2.56", "reportTime": "pre-market"},
            {"fiscalDateEnding": "2025-12-31", "reportedDate": "2026-01-28", "reportedEPS": "2.0", "estimatedEPS": "1.8",
             "surprise": "None", "surprisePercentage": "None", "reportTime": "post-market"},
            {"fiscalDateEnding": "2025-09-30", "reportedDate": "2025-10-22", "reportedEPS": "1.7", "estimatedEPS": "1.6",
             "surprise": "0.1", "surprisePercentage": "6.25", "reportTime": "post-market"},
            {"fiscalDateEnding": "2025-06-30", "reportedDate": "2025-07-23", "reportedEPS": "1.6", "estimatedEPS": "1.5",
             "surprise": "0.1", "surprisePercentage": "6.6", "reportTime": "post-market"}]}
CAL = ("symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
       "AAA,Alpha Inc,2026-10-22,2026-09-30,2.15,USD,post-market\n"
       "BRK.B,Berkshire,2026-11-01,2026-09-30,,USD,\n"
       "ZZZZ,Not In Universe,2026-10-25,2026-09-30,0.5,USD,pre-market\n")
NEWS = {"items": "3", "sentiment_score_definition": "...", "relevance_score_definition": "...", "feed": [
    {"title": "Alpha beats", "url": "https://x/1", "time_published": "20261016T183000", "source": "Wire",
     "ticker_sentiment": [{"ticker": "AAA", "relevance_score": "0.9", "ticker_sentiment_score": "0.41"},
                          {"ticker": "CRYPTO:BTC", "relevance_score": "0.9", "ticker_sentiment_score": "0.2"},
                          {"ticker": "BBB", "relevance_score": "0.05", "ticker_sentiment_score": "-0.3"}]}]}
UST = {"name": "10-Year Treasury", "interval": "daily", "unit": "percent",
       "data": [{"date": "2026-10-16", "value": "4.12"}, {"date": "2026-10-15", "value": "."}]}
RATE = {"Information": "We have detected your API key as TESTKEY123456789 and our standard API rate limit is 25 requests per day."}
PREMIUM = {"Information": "Thank you for using Alpha Vantage! This is a premium endpoint."}


def fake_http(responses: dict, log: list | None = None):
    def get(url, params):
        if log is not None:
            log.append(dict(params))
        fn = params["function"]
        r = responses.get((fn, params.get("symbol"))) or responses.get(fn)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)
    return get


def _cfg(**over):
    cfg = make_cfg()
    cfg.setdefault("alpha_vantage", {}).update({"api_key": KEY, "enabled": True, "daily_limit": 25, "per_minute_limit": 5,
                                                "news_calls_per_run": 2, "treasury_maturities": ["10year"]})
    cfg["alpha_vantage"].update(over)
    return cfg


def _ctx_with_stocks(cfg, tickers=("AAA", "BBB", "BRK-B", "CCC")):
    ctx = make_ctx(cfg)
    ctx.repo.upsert_stocks(pd.DataFrame({"ticker": list(tickers), "name": list(tickers), "sector": "X", "subsector": None,
                                         "board": "S&P 500", "is_active": True}))
    return ctx


class TestParsers(unittest.TestCase):
    def test_earnings(self):
        e = parse_earnings(EARN, "AAA")
        self.assertEqual(len(e), 5)
        row = e[e["fiscal_date_ending"] == "2025-12-31"].iloc[0]
        self.assertAlmostEqual(row["surprise"], 0.2)            # "None" → dihitung dari EPS − estimasi
        self.assertEqual(e.iloc[0]["report_time"], "post-market")

    def test_calendar_normalizes_tickers(self):
        c = parse_calendar(CAL)
        self.assertEqual(c["ticker"].tolist(), ["AAA", "BRK-B", "ZZZZ"])
        self.assertTrue(np.isnan(c.loc[c["ticker"] == "BRK-B", "estimate"].iloc[0]))
        self.assertEqual(c.loc[0, "time_of_day"], "post-market")
        with self.assertRaises(AlphaVantageError):
            parse_calendar("foo,bar\n1,2\n")

    def test_news_filters_relevance_and_non_equity(self):
        n = parse_news(NEWS, 0.25)
        self.assertEqual(n["ticker"].tolist(), ["AAA"])
        self.assertAlmostEqual(n["sentiment"].iloc[0], 0.41)

    def test_series_skips_missing(self):
        s = parse_series(UST, "UST10YEAR")
        self.assertEqual(len(s), 1)
        self.assertAlmostEqual(s["value"].iloc[0], 4.12)


class TestClient(unittest.TestCase):
    def test_rate_limit_stops_and_key_never_leaks(self):
        cfg = _cfg()
        av = AlphaVantageClient(cfg, None, http_get=fake_http({"EARNINGS": RATE}), sleep=lambda s: None)
        with self.assertRaises(QuotaExhausted) as cm:
            av.earnings("AAA")
        self.assertNotIn(KEY, str(cm.exception))
        self.assertEqual(av.remaining(), 0)

    def test_premium_detected(self):
        av = AlphaVantageClient(_cfg(), None, http_get=fake_http({"EARNINGS": PREMIUM}), sleep=lambda s: None)
        with self.assertRaises(PremiumOnly):
            av.earnings("AAA")

    def test_network_error_redacted(self):
        boom = AlphaVantageError("ConnectionError: https://www.alphavantage.co/query?function=X&apikey=" + KEY)
        av = AlphaVantageClient(_cfg(), None, http_get=fake_http({"EARNINGS": boom}), sleep=lambda s: None)
        with self.assertRaises(AlphaVantageError):
            av.earnings("AAA")
        from app.data.alpha_vantage import _redact
        self.assertNotIn(KEY, _redact(str(boom), KEY))

    def test_throttle_5_per_minute(self):
        waits = []
        av = AlphaVantageClient(_cfg(), None, http_get=fake_http({"EARNINGS": EARN}), sleep=waits.append)
        for _ in range(3):
            av.earnings("AAA")
        self.assertEqual(len(waits), 2)
        self.assertTrue(all(w > 11 for w in waits))

    def test_not_configured_without_key(self):
        cfg = make_cfg()
        cfg.setdefault("alpha_vantage", {})["api_key"] = ""
        with mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": ""}):
            self.assertFalse(AlphaVantageClient(cfg).configured())


class TestUpdate(unittest.TestCase):
    def _run(self, cfg, ctx, responses, today="2026-10-17", mode="daily", log=None, **kw):
        from app.data.alpha_vantage_update import update_alpha_vantage
        av = AlphaVantageClient(cfg, ctx.repo, http_get=fake_http(responses, log), sleep=lambda s: None)
        return update_alpha_vantage(cfg, ctx.repo, today, mode=mode, client=av, **kw)

    def test_daily_run_fills_everything_within_budget(self):
        cfg = _cfg()
        ctx = _ctx_with_stocks(cfg)
        calls = []
        out = self._run(cfg, ctx, {"EARNINGS_CALENDAR": CAL, "TREASURY_YIELD": UST, "FEDERAL_FUNDS_RATE": UST,
                                   "NEWS_SENTIMENT": NEWS, "EARNINGS": EARN}, log=calls)
        self.assertEqual(out["calendar_rows"], 2)                      # ZZZZ di luar universe tidak disimpan
        self.assertEqual(out["earnings_tickers"], 4)                   # semua saham universe di-backfill
        self.assertLessEqual(out["calls"], 25)
        self.assertGreater(out["news_rows"], 0)
        self.assertTrue(all("apikey" in c for c in calls))
        from app.data.alpha_vantage_update import apply_market_rates, risk_free_from_treasury
        rf, src = risk_free_from_treasury(cfg, ctx.repo, "2026-10-17")
        self.assertAlmostEqual(rf, 0.0412)
        self.assertIn("Treasury 10Y", apply_market_rates(cfg, ctx.repo, "2026-10-17"))
        self.assertIsNone(risk_free_from_treasury(cfg, ctx.repo, "2026-12-31")[0])    # basi → tidak dipakai
        self.assertIsNone(risk_free_from_treasury(cfg, ctx.repo, "2026-10-10")[0])    # PIT: belum terbit

    def test_budget_is_shared_across_runs(self):
        cfg = _cfg(daily_limit=6)
        ctx = _ctx_with_stocks(cfg)
        r = {"EARNINGS_CALENDAR": CAL, "TREASURY_YIELD": UST, "FEDERAL_FUNDS_RATE": UST, "NEWS_SENTIMENT": NEWS,
             "EARNINGS": EARN}
        first = self._run(cfg, ctx, r)
        self.assertEqual(first["calls"], 6)
        second = self._run(cfg, ctx, r, mode="backfill")
        self.assertEqual(second["calls"], 0)                           # jendela 24 jam sudah penuh

    def test_calendar_change_log(self):
        cfg = _cfg(news_calls_per_run=0, treasury_maturities=[])
        ctx = _ctx_with_stocks(cfg)
        from app.data.alpha_vantage_update import load_calendar, store_calendar
        c = parse_calendar(CAL)
        self.assertEqual(store_calendar(ctx.repo, c, "2026-10-16"), 2)
        self.assertEqual(store_calendar(ctx.repo, c, "2026-10-17"), 0)          # tidak berubah → tidak ditulis
        c2 = c.copy()
        c2.loc[c2["ticker"] == "AAA", "estimate"] = 2.20                        # revisi konsensus → baris baru
        self.assertEqual(store_calendar(ctx.repo, c2, "2026-10-18"), 1)
        self.assertEqual(len(load_calendar(ctx.repo)), 3)

    def test_queue_puts_fresh_reports_first(self):
        cfg = _cfg()
        ctx = _ctx_with_stocks(cfg)
        from app.data.alpha_vantage_update import earnings_queue
        from app.valuation.fundamental_provider import store_earnings_events
        store_earnings_events(ctx.repo, pd.DataFrame({"ticker": ["CCC"], "accession": ["x1"], "filing_date": ["2026-10-14"],
                                                      "accepted_at": [None], "reaction_date": ["2026-10-15"],
                                                      "form": ["8-K"]}))
        q = earnings_queue(ctx.repo, cfg, "2026-10-17")
        self.assertEqual(q[0], "CCC")
        self.assertEqual(sorted(q), ["AAA", "BBB", "BRK-B", "CCC"])

    def test_without_key_is_skipped_cleanly(self):
        cfg = make_cfg()
        cfg.setdefault("alpha_vantage", {})["api_key"] = ""
        ctx = _ctx_with_stocks(cfg)
        from app.data.alpha_vantage_update import update_alpha_vantage
        with mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": ""}):
            out = update_alpha_vantage(cfg, ctx.repo, "2026-10-17")
        self.assertFalse(out["configured"])
        self.assertEqual(out["calls"], 0)


class TestSurpriseFeatures(unittest.TestCase):
    def _panel(self):
        d = pd.bdate_range("2025-06-02", "2026-11-30")
        return pd.DataFrame({"ticker": "AAA", "date": d})

    def test_point_in_time_and_sue(self):
        from app.features.surprise import add_surprise_features
        sur = parse_earnings(EARN, "AAA")
        out = add_surprise_features(self._panel(), sur).set_index("date")
        self.assertTrue(np.isnan(out.loc["2026-07-23", "f_eps_surprise_pct"]))      # post-market → belum diketahui
        self.assertAlmostEqual(out.loc["2026-07-24", "f_eps_surprise_pct"], 0.05)   # hari bursa berikutnya
        self.assertAlmostEqual(out.loc["2026-04-22", "f_eps_surprise_pct"], -0.05 / 1.95)   # pre-market → hari itu
        self.assertTrue(np.isfinite(out.loc["2026-07-24", "f_sue"]))                # ≥ 4 kuartal riwayat
        self.assertEqual(out.loc["2026-07-24", "f_beat_streak"], 1)
        self.assertEqual(out.loc["2026-04-22", "f_beat_streak"], 0)
        self.assertTrue(np.isfinite(out.loc["2026-10-16", "f_sue"]))                # 84 hari → masih berlaku
        self.assertTrue(np.isnan(out.loc["2026-10-26", "f_sue"]))                   # > 91 hari → kedaluwarsa

    def test_calendar_overrides_estimate_only_where_known(self):
        from app.features.surprise import apply_earnings_calendar
        df = pd.DataFrame({"ticker": "AAA", "date": pd.to_datetime(["2026-10-01", "2026-10-16"]),
                           "earn_soon_days": [30.0, 15.0]})
        cal = parse_calendar(CAL).assign(as_of=pd.Timestamp("2026-10-15"))
        out = apply_earnings_calendar(df, cal)
        self.assertEqual(out["earn_soon_days"].tolist(), [30.0, 6.0])               # 16 → 22 Okt = 6 hari
        self.assertEqual(out["earn_date_source"].tolist(), ["estimate", "calendar"])


class TestNewsCoverage(unittest.TestCase):
    def test_counts_unknown_before_news_history(self):
        from app.features.fundamental_sentiment import merge_news
        d = pd.bdate_range("2026-08-03", "2026-10-16")
        df = pd.DataFrame({"ticker": "AAA", "date": d})
        news = pd.DataFrame({"ticker": ["AAA"], "published_at": [pd.Timestamp("2026-10-01 10:00")], "sentiment": [0.4]})
        out = merge_news(df, news).set_index("date")
        self.assertTrue(np.isnan(out.loc["2026-08-10", "news_count5"]))
        self.assertEqual(out.loc["2026-10-16", "news_count5"], 0)


class TestAnnotations(unittest.TestCase):
    def test_only_inside_actions(self):
        from app.utils.gha import annotate
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": ""}), mock.patch("builtins.print") as p:
            annotate("x", "y")
            p.assert_not_called()
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), mock.patch("builtins.print") as p:
            annotate("A, B", "line1\nline2")
            msg = p.call_args[0][0]
            self.assertTrue(msg.startswith("::notice title=A%2C B::line1%0Aline2"))


if __name__ == "__main__":
    unittest.main()
