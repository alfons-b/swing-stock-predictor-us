import unittest

import numpy as np
import pandas as pd

from app.strategy.take_profit import compute_targets
from tests.helpers import make_cfg


def targets(res50=np.nan):
    df = pd.DataFrame({"res20": [np.nan], "res50": [res50], "res120": [np.nan]})
    return compute_targets(df, pd.DataFrame({"entry_ideal": [1250.0]}), pd.DataFrame({"stop_loss": [1190.0]}), make_cfg()).iloc[0]


class TestTakeProfit(unittest.TestCase):
    def test_rr_multiples(self):
        t = targets()
        self.assertEqual((t["take_profit_1"], t["take_profit_2"]), (1340.0, 1400.0))   # 1.5R, 2.5R
        self.assertEqual(t["tp_warning"], "")

    def test_resistance_between_tp1_tp2(self):
        t = targets(1380.0)
        self.assertLess(t["take_profit_2"], 1380.0)
        self.assertIn("TP2", t["tp_warning"])

    def test_resistance_below_tp1(self):
        t = targets(1290.0)
        self.assertLess(t["take_profit_1"], 1290.0)
        self.assertIn("TP1", t["tp_warning"])


if __name__ == "__main__":
    unittest.main()
