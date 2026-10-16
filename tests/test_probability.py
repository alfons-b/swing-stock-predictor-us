import unittest

import numpy as np

from app.models.evaluation import calibration_table, expected_calibration_error
from tests.test_model import fit


class TestProbability(unittest.TestCase):
    def test_probabilities_valid(self):
        _, _, p = fit(1.0)
        P = p[["probability_bearish", "probability_neutral", "probability_bullish"]].to_numpy()
        self.assertTrue(((P >= 0) & (P <= 1)).all())
        self.assertTrue(np.allclose(P.sum(axis=1), 1, atol=1e-6))

    def test_calibration_reasonable(self):
        _, te, p = fit(1.0)
        ece = expected_calibration_error((te["y_class"] == 2).astype(int).to_numpy(), p["probability_bullish"].to_numpy())
        self.assertLess(ece, 0.05)

    def test_ece_detects_miscalibration(self):
        y = np.random.default_rng(0).binomial(1, 0.2, 5000)
        self.assertGreater(expected_calibration_error(y, np.full(5000, 0.8)), 0.5)
        self.assertGreater(len(calibration_table(y, np.random.default_rng(1).uniform(size=5000))), 5)


if __name__ == "__main__":
    unittest.main()
