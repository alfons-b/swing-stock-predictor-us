import unittest

from app.data.cleaner import clean_prices
from app.data.providers.csv_provider import CSVProvider
from app.data.validator import issues_to_frame, validate_prices
from tests.helpers import make_cfg


class TestDataValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_cfg()
        p = CSVProvider(cls.cfg["market_data"]["providers"][0], cls.cfg)
        cls.u = p.get_universe()
        cls.raw = p._all_prices().pipe(lambda d: __import__("app.data.schema", fromlist=["x"]).conform_prices(d))
        import pandas as pd
        cls.ca = pd.concat([p.get_actions(t, None, None) for t in cls.u["ticker"]], ignore_index=True)
        cls.rep = issues_to_frame(validate_prices(cls.raw, cls.u, cls.ca, cls.cfg)).set_index("check")
        cls.clean, _ = clean_prices(cls.raw, cls.u, cls.ca, cls.cfg)

    def test_detects_planted_problems(self):
        for c in ("missing_ohlcv", "duplicate_date", "invalid_price", "suspension_or_halt", "split_or_reverse_split_unadjusted"):
            self.assertGreater(self.rep.loc[c, "count"], 0, c)

    def test_clean_output_valid(self):
        c = self.clean
        self.assertFalse(c.duplicated(["ticker", "date"]).any())
        self.assertFalse(c[["open", "high", "low", "close", "volume"]].isna().any().any())
        self.assertTrue((c["high"] >= c["low"]).all())

    def test_split_adjusted(self):
        s = self.clean[self.clean["ticker"] == "ZSPL"].sort_values("date")
        self.assertLess(s["close"].pct_change().abs().max(), 0.35)

    def test_suspension_flagged(self):
        self.assertGreaterEqual(int(self.clean.loc[self.clean["ticker"] == "ZSUS", "is_suspended"].sum()), 20)


if __name__ == "__main__":
    unittest.main()
