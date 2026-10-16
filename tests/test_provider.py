import unittest

import pandas as pd

from app.data.providers.base import MarketDataProvider, ProviderUnavailable
from app.data.providers.chain import ProviderChain
from app.data.providers.csv_provider import CSVProvider
from app.data.tickers import normalize_ticker, to_provider_symbol
from tests.helpers import make_cfg


class Flaky(MarketDataProvider):
    type = "fake"

    def __init__(self, cfg, fail_times=1, bad=("BBBB",), unavailable=False, prio=1):
        super().__init__({"name": f"flaky{prio}", "priority": prio}, cfg)
        self.calls, self.fail_times, self.bad, self.unavailable = 0, fail_times, set(bad), unavailable

    def get_prices_batch(self, tickers, start, end):
        self.calls += 1
        if self.unavailable:
            raise ProviderUnavailable("no credentials")
        if self.calls <= self.fail_times:
            raise ConnectionError("timeout")
        d = pd.bdate_range(start, end)
        return {t: pd.DataFrame({"ticker": t, "date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
                for t in tickers if t not in self.bad}


class TestProviders(unittest.TestCase):
    def test_ticker_normalization(self):
        self.assertEqual(normalize_ticker(" aapl "), "AAPL")
        self.assertEqual(normalize_ticker("NASDAQ:MSFT"), "MSFT")
        for raw in ("BRK.B", "brk/b", "BRK B", "NYSE:BRK.B", "$BRK-B"):
            self.assertEqual(normalize_ticker(raw), "BRK-B")
        self.assertEqual(normalize_ticker("AAPL.US"), "AAPL")
        self.assertEqual(to_provider_symbol("bf.b"), "BF-B")
        for bad in ("", "123", "TOOLONGX", "BRK-BBB"):
            with self.assertRaises(ValueError):
                normalize_ticker(bad)

    def test_csv_provider_respects_as_of(self):
        cfg = make_cfg(as_of="2019-06-28")
        p = CSVProvider(cfg["market_data"]["providers"][0], cfg)
        self.assertLessEqual(p.get_index(None, None)["date"].max(), pd.Timestamp("2019-06-28"))
        tk = p.get_universe()["ticker"].iloc[0]
        self.assertLessEqual(p.get_prices(tk, "2019-01-01", None)["date"].max(), pd.Timestamp("2019-06-28"))

    def test_retry_with_backoff_then_success(self):
        cfg = make_cfg()
        waits = []
        prov = Flaky(cfg, fail_times=2, bad=())
        chain = ProviderChain(cfg, [prov], sleep=waits.append)
        chain.attempts = 3
        chain.backoff = 2
        res = chain.fetch_prices({"AAAA": pd.Timestamp("2024-01-01")}, pd.Timestamp("2024-01-05"))
        self.assertIn("AAAA", res.frames)
        self.assertEqual(waits, [2, 4])  # exponential backoff

    def test_one_ticker_failure_does_not_abort_and_fallback(self):
        cfg = make_cfg()
        primary = Flaky(cfg, fail_times=0, bad=("NVDA",), prio=1)
        backup = Flaky(cfg, fail_times=0, bad=(), prio=2)
        chain = ProviderChain(cfg, [primary, backup], sleep=lambda s: None)
        req = {t: pd.Timestamp("2024-01-01") for t in ("AAPL", "MSFT", "NVDA", "JPM")}
        res = chain.fetch_prices(req, pd.Timestamp("2024-01-03"))
        self.assertEqual(set(res.frames), set(req))
        self.assertEqual(res.sources["NVDA"], "flaky2")       # fallback provider
        self.assertEqual(res.sources["AAPL"], "flaky1")

    def test_unavailable_provider_skipped_without_retry(self):
        cfg = make_cfg()
        idx = Flaky(cfg, unavailable=True, prio=1)
        ok = Flaky(cfg, fail_times=0, bad=(), prio=2)
        res = ProviderChain(cfg, [idx, ok], sleep=lambda s: self.fail("tidak boleh retry")).fetch_prices(
            {"AAAA": pd.Timestamp("2024-01-01")}, pd.Timestamp("2024-01-02"))
        self.assertEqual(idx.calls, 1)
        self.assertIn("AAAA", res.frames)

    def test_yahoo_defaults_are_us(self):
        from app.data.providers.yahoo_provider import YahooFinanceProvider
        y = YahooFinanceProvider({"name": "y"}, make_cfg())
        self.assertEqual((y.suffix, y.index_symbol), ("", "^GSPC"))


if __name__ == "__main__":
    unittest.main()


class TestProviderAvailability(unittest.TestCase):
    def test_csv_without_data_is_skipped(self):
        """Regresi: cadangan CSV tanpa folder data di runner cloud dulu menghasilkan 41 'gagal' + PARTIAL_SUCCESS."""
        from app.data.providers.chain import make_providers
        cfg = make_cfg()
        cfg["market_data"]["providers"] = [dict(cfg["market_data"]["providers"][0], dir="/tidak/ada")]
        self.assertEqual(make_providers(cfg), [])

    def test_no_data_status_not_overwritten_by_fallback_error(self):
        cfg = make_cfg()
        empty = Flaky(cfg, fail_times=0, bad=("ARMY",), prio=1)        # provider merespons, tapi tanpa data ARMY
        broken = Flaky(cfg, fail_times=99, bad=(), prio=2)              # cadangan yang error
        res = ProviderChain(cfg, [empty, broken], sleep=lambda s: None).fetch_prices(
            {"ARMY": pd.Timestamp("2024-01-01"), "AAPL": pd.Timestamp("2024-01-01")}, pd.Timestamp("2024-01-03"))
        self.assertIn("AAPL", res.frames)
        self.assertTrue(res.failed["ARMY"].endswith("tidak ada data baru"))
