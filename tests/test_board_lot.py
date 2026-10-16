"""Aturan eksekusi pasar AS: tick $0,01 (≥ $1) / $0,0001 (< $1), tanpa lot (kelipatan 1 saham)."""
import unittest

import numpy as np

from app.utils.market_rules import BOARD_LOT, round_to_tick, shares_to_lots, tick_size
from tests.test_position_sizing import size


class TestUSOrderRules(unittest.TestCase):
    def test_position_is_whole_shares(self):
        self.assertEqual(BOARD_LOT, 1)
        for e, s in ((123.45, 118.0), (8.70, 8.05), (512.5, 490.0), (31.2, 29.6)):
            r = size(e, s)
            self.assertEqual(r["position_size"], int(r["position_size"]))
            self.assertGreaterEqual(r["position_size"], 0)

    def test_tick_size_rules(self):
        self.assertEqual([tick_size(p) for p in (0.5, 0.9999, 1.0, 45.0, 950.0)], [0.0001, 0.0001, 0.01, 0.01, 0.01])
        for p in np.random.default_rng(0).uniform(1.0, 2000, 300):
            for mode in ("down", "up", "nearest"):
                v = round_to_tick(p, mode)
                self.assertAlmostEqual(v * 100, round(v * 100), places=6)        # tepat kelipatan $0,01
                if mode == "down":
                    self.assertLessEqual(v, p + 1e-9)
                if mode == "up":
                    self.assertGreaterEqual(v, p - 1e-9)
        self.assertEqual(round_to_tick(12.345, "down"), 12.34)
        self.assertEqual(round_to_tick(12.341, "up"), 12.35)
        self.assertEqual(round_to_tick(0.12346, "nearest"), 0.1235)             # sub-dollar: 4 desimal
        self.assertEqual(round_to_tick(0.99996, "up"), 1.0)                     # pindah band tick
        self.assertEqual(shares_to_lots(1999.7), 1999)


if __name__ == "__main__":
    unittest.main()
