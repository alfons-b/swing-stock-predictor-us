import unittest

import pandas as pd

from app.models.labels import add_labels
from tests.helpers import make_cfg


def frame(closes, highs=None, lows=None, atr=10.0):
    n = len(closes)
    return pd.DataFrame({"ticker": "AAAA", "date": pd.bdate_range("2024-01-01", periods=n), "close": closes,
                         "high": highs or closes, "low": lows or closes, "atr14": atr, "vol20": 0.02})


class TestTargets(unittest.TestCase):
    def test_forward_return_and_classes(self):
        cfg = make_cfg()
        closes = [100.0] * 3 + [110.0] * 20
        df = add_labels(frame(closes), cfg)
        self.assertAlmostEqual(df["y_ret_5d"].iloc[0], 0.10)
        self.assertEqual(df["y_class"].iloc[0], 2)                  # > +3% → bullish
        self.assertEqual(df["y_class"].iloc[5], 1)                  # flat → neutral
        self.assertTrue(df["y_ret_5d"].tail(5).isna().all())        # tidak ada masa depan → NaN (tidak ditebak)
        self.assertTrue(df["y_class"].tail(5).isna().all())

    def test_hit_tp_before_sl_is_pessimistic(self):
        cfg = make_cfg()
        closes = [100.0] * 12
        highs = [100.0, 125.0] + [100.0] * 10                       # TP (100+2*10) & SL (100-1.5*10) di candle sama
        lows = [100.0, 80.0] + [100.0] * 10
        df = add_labels(frame(closes, highs, lows), cfg)
        self.assertEqual(df["y_hit_tp"].iloc[0], 0)                 # candle sama → SL dulu
        self.assertEqual(df["y_hit_sl_first"].iloc[0], 1)

    def test_mfe_mae(self):
        cfg = make_cfg()
        closes = [100.0] * 12
        highs = [100, 104, 108, 101, 100, 100, 100, 100, 100, 100, 100, 100]
        lows = [100, 99, 97, 98, 100, 100, 100, 100, 100, 100, 100, 100]
        df = add_labels(frame(closes, [float(x) for x in highs], [float(x) for x in lows]), cfg)
        self.assertAlmostEqual(df["y_mfe_5d"].iloc[0], 0.08)
        self.assertAlmostEqual(df["y_mae_5d"].iloc[0], -0.03)


if __name__ == "__main__":
    unittest.main()
