import unittest
from types import SimpleNamespace

import pandas as pd

from app.pipeline.evaluate import evaluate_one
from tests.helpers import make_cfg


def pred(**kw):
    base = dict(id=1, close_price=1000.0, entry_high=1010.0, stop_loss=950.0, tp1=1075.0, tp2=1125.0,
                prob_bearish=0.2, prob_neutral=0.3, prob_bullish=0.5)
    base.update(kw)
    return SimpleNamespace(**base)


def bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["date"] = pd.bdate_range("2024-01-02", periods=len(df))
    return df


class TestPredictionEvaluation(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()

    def test_waits_for_horizon(self):
        self.assertIsNone(evaluate_one(pred(), bars([(1000, 1010, 990, 1000)] * 5), self.cfg))

    def test_tp2_hit_and_correct(self):
        rows = [(1000, 1005, 995, 1000), (1010, 1080, 1005, 1070), (1070, 1130, 1060, 1120)] + [(1120, 1125, 1110, 1120)] * 8
        r = evaluate_one(pred(), bars(rows), self.cfg)
        self.assertEqual((r["entry_filled"], r["hit_tp1"], r["hit_tp2"], r["hit_stop"], r["outcome"]), (True, True, True, False, "TP2"))
        self.assertAlmostEqual(r["actual_return_5d"], 0.12)
        self.assertEqual(r["actual_class"], 2)
        self.assertTrue(r["prediction_correct"])
        self.assertAlmostEqual(r["mfe"], 0.13)

    def test_stop_first_and_wrong(self):
        rows = [(1000, 1005, 995, 1000), (990, 1200, 940, 960)] + [(950, 955, 940, 945)] * 9
        r = evaluate_one(pred(), bars(rows), self.cfg)
        self.assertEqual((r["hit_stop"], r["hit_tp1"], r["outcome"]), (True, False, "STOP"))
        self.assertFalse(r["prediction_correct"])
        self.assertLess(r["mae"], -0.05)

    def test_not_filled(self):
        rows = [(1050, 1060, 1040, 1050)] * 11
        r = evaluate_one(pred(), bars(rows), self.cfg)
        self.assertEqual((r["entry_filled"], r["outcome"]), (False, "NOT_FILLED"))


if __name__ == "__main__":
    unittest.main()
