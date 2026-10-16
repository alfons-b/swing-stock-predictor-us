import unittest

import pandas as pd

from app.risk.position_sizing import compute_position_size
from tests.helpers import make_cfg


def size(entry, stop, regime="BULL", adv=500e6, pv=10_000):
    df = pd.DataFrame({"market_regime": [regime], "avg_value20": [adv]})
    return compute_position_size(df, pd.DataFrame({"entry_ideal": [entry]}), pd.DataFrame({"stop_loss": [stop]}), make_cfg(), pv).iloc[0]


class TestPositionSizing(unittest.TestCase):
    def test_risk_based(self):
        s = size(100, 95)  # 1% × $10.000 = $100 / $5 = 20 saham; maks 20% portofolio = $2.000 / $100 = 20 saham
        self.assertEqual(s["position_size"], 20)
        self.assertLessEqual(s["risk_amount"], 100 + 1e-6)
        self.assertAlmostEqual(s["capital_required"], 20 * 100)
        self.assertAlmostEqual(s["estimated_loss"], 20 * 5)

    def test_regime_reduces_size(self):
        self.assertGreater(size(100, 90)["position_size"], size(100, 90, "STRONG_BEAR")["position_size"])

    def test_liquidity_cap(self):
        self.assertLessEqual(size(50, 48, adv=1e6, pv=1e7)["capital_required"], 0.05 * 1e6 + 1)


if __name__ == "__main__":
    unittest.main()
