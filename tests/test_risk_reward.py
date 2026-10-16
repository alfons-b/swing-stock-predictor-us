import unittest

import pandas as pd

from app.scanner.ranking import rank_signals
from tests.helpers import make_cfg
from tests.test_take_profit import targets


class TestRiskReward(unittest.TestCase):
    def test_blended_rr(self):
        self.assertAlmostEqual(targets()["risk_reward"], 2.0)       # 0.5*1.5 + 0.5*2.5

    def test_rr_only_to_resistance_when_blocked(self):
        t = targets(1290.0)
        self.assertLess(t["risk_reward"], make_cfg()["strategy"]["MIN_RISK_REWARD"])  # → ditolak filter RR

    def test_rank_signals_empty_no_crash_with_string_dtype(self):
        """Regresi bug Windows (pandas + pyarrow): tidak ada BUY → map() pada Series kosong ber-dtype string."""
        cfg = make_cfg()
        sig = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 2), "ticker": ["AAAA", "BBBB"],
                            "decision": pd.Series(["WAIT", "AVOID"], dtype="string"),
                            "market_regime": pd.Series(["BULL", "BULL"], dtype="string"), "rank_score": [50.0, 40.0]})
        self.assertEqual(len(rank_signals(sig, cfg)), 0)
        sig.loc[0, "decision"] = "BUY"
        out = rank_signals(sig, cfg)
        self.assertEqual(out["ticker"].tolist(), ["AAAA"])


if __name__ == "__main__":
    unittest.main()
