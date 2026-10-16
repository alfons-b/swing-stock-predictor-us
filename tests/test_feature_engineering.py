import unittest

import numpy as np
import pandas as pd

from app.features import indicators as ind
from app.features.builder import build_features, feature_columns
from tests.test_no_data_leakage import load_market


class TestIndicators(unittest.TestCase):
    def test_basic_indicators(self):
        s = pd.Series(np.arange(1, 61, dtype=float))
        self.assertAlmostEqual(ind.sma(s, 5).iloc[-1], 58.0)
        self.assertTrue(np.isnan(ind.sma(s, 5).iloc[3]))
        self.assertEqual(ind.rsi(s, 14).iloc[-1], 100.0)        # naik terus → RSI 100
        h, l, c = s + 1, s - 1, s
        self.assertAlmostEqual(ind.atr(h, l, c, 14).iloc[-1], 2.0, places=6)

    def test_feature_set(self):
        cfg = __import__("tests.helpers", fromlist=["x"]).make_cfg()
        df = build_features(load_market(cfg), cfg)
        f = feature_columns(df)
        for name in ("f_dist_sma200", "f_rsi14", "f_adx", "f_macd_hist_slope", "f_atr_pct", "f_volume_ratio", "f_rs20",
                     "f_rs_sector20", "f_regime_score", "f_sector_score", "f_ret60", "f_bb_width"):
            self.assertIn(name, f)
        for raw in ("sma5", "sma100", "ema9", "res20", "higher_high", "lower_low", "obv" if "obv" in df else "f_obv_slope"):
            self.assertIn(raw, df.columns)
        self.assertTrue(set(df["market_regime"]) <= {"STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"})


if __name__ == "__main__":
    unittest.main()


def suspended_then_active(n_live=400, n_frozen=2200, seed=1):
    """Saham diperdagangkan, lalu suspensi ~9 tahun (harga beku, volume 0), lalu aktif lagi."""
    rng = np.random.default_rng(seed)
    live = 100 * np.cumprod(1 + rng.normal(0, .02, n_live))
    c = np.r_[live, np.full(n_frozen, live[-1]), live[-1] * np.array([1.0, 1.02, 1.03, 1.01, 1.05])]
    v = np.r_[rng.uniform(1e5, 1e6, n_live), np.zeros(n_frozen), rng.uniform(1e5, 1e6, 5)]
    return pd.DataFrame({"ticker": "SUSP", "date": pd.bdate_range("2015-01-01", periods=len(c)), "open": c, "high": c,
                         "low": c, "close": c, "volume": v, "value": c * v})


class TestExtremeValues(unittest.TestCase):
    """Regresi (setup production): ATR → 1e-72 saat suspensi panjang membuat f_atr_contraction = 3e69 > float32."""

    def test_long_suspension_features_float32_safe(self):
        from app.features.technical import compute_stock_features
        from tests.helpers import make_cfg
        f = compute_stock_features(suspended_then_active(), make_cfg())
        X = f[[c for c in f.columns if c.startswith("f_")]].replace([np.inf, -np.inf], np.nan)
        self.assertLess(float(X.abs().max().max()), 1e7)

    def test_sanitize_clips_and_removes_inf(self):
        from app.features.builder import FEATURE_ABS_MAX, sanitize_features
        df = sanitize_features(pd.DataFrame({"f_a": [1.0, np.inf, -1e70], "close": [1e70, 1, 1]}))
        self.assertTrue(np.isnan(df["f_a"].iloc[1]))
        self.assertEqual(df["f_a"].iloc[2], -FEATURE_ABS_MAX)
        self.assertEqual(df["close"].iloc[0], 1e70)                 # kolom non-fitur tidak disentuh

    def test_random_forest_predicts_extreme_rows(self):
        from app.models.trainer import fit_swing_model
        from tests.helpers import make_cfg
        from tests.test_model import panel
        cfg = make_cfg()
        df = panel(1.0, n_days=200, n_tk=30)
        m = fit_swing_model(df, [f"f_{i}" for i in range(6)], cfg, ["random_forest"], with_aux=False)
        bad = df.head(3).copy()
        bad.loc[bad.index[0], "f_0"], bad.loc[bad.index[1], "f_1"] = 3.3e69, np.inf
        p = m.predict(bad)                                          # dulu: ValueError float32
        self.assertTrue(np.isfinite(p["probability_bullish"]).all())
