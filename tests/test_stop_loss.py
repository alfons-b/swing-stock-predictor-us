import unittest

import pandas as pd

from app.strategy.stop_loss import compute_stops
from tests.helpers import make_cfg


def run(setup, inval, swing=900.0, atr=20.0, entry=1000.0):
    df = pd.DataFrame({"atr14": [atr], "swing_low10": [swing]})
    st = pd.DataFrame({"setup_type": [setup], "invalidation_level": [inval]})
    return compute_stops(df, st, pd.DataFrame({"entry_ideal": [entry]}), make_cfg()).iloc[0]


class TestStopLoss(unittest.TestCase):
    def test_method_by_setup(self):
        self.assertEqual(run("BREAKOUT", 960.0)["stop_method"], "support")
        self.assertEqual(run("EMA20_PULLBACK", 960.0, swing=950.0)["stop_method"], "swing_low")
        self.assertEqual(run("TREND_CONTINUATION", 960.0)["stop_method"], "atr")
        self.assertEqual(run("TREND_CONTINUATION", 960.0)["stop_loss"], 970.0)   # 1000 - 1.5*20

    def test_min_distance_and_too_wide(self):
        r = run("BREAKOUT", 998.0, atr=1.0)                   # stop terlalu dekat → dilebarkan ke min 2%
        self.assertLessEqual(r["stop_loss"], 980.0)
        self.assertIn("min_dist", r["stop_method"])
        self.assertTrue(run("BREAKOUT", 700.0)["stop_too_wide"])

    def test_fallback_when_structure_missing(self):
        self.assertEqual(run("BREAKOUT", float("nan"))["stop_method"], "atr")


if __name__ == "__main__":
    unittest.main()
