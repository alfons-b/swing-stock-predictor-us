"""Model (§29) + persistensi & versioning (§13, §14) + kebijakan promosi (§15)."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.models.retrain import evaluate_gates
from app.models.trainer import fit_swing_model
from app.storage.model_storage import DatabaseModelStorage, LocalModelStorage, ModelStorageError
from tests.helpers import make_cfg, make_ctx


def panel(signal, seed=0, n_days=400, n_tk=40):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    X = rng.normal(size=(n_days * n_tk, 6))
    ret = signal * 0.02 * X[:, 0] + 0.04 * rng.normal(size=len(X))
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(6)])
    df["date"] = np.repeat(dates, n_tk)
    for h in (3, 5, 10):
        df[f"y_ret_{h}d"] = ret
    df["y_class"] = np.where(ret > 0.03, 2, np.where(ret < -0.03, 0, 1))
    df["y_mfe_5d"], df["y_mae_5d"] = np.abs(ret), -np.abs(ret)
    df["y_hit_tp"], df["y_hit_sl_first"] = (ret > 0.03).astype(float), (ret < -0.03).astype(float)
    return df


def fit(signal, cfg=None):
    cfg = cfg or make_cfg()
    df = panel(signal)
    cut = df["date"].quantile(0.7)
    tr, te = df[df["date"] <= cut], df[df["date"] > cut]
    m = fit_swing_model(tr, [f"f_{i}" for i in range(6)], cfg, ["logistic"])
    return m, te, m.predict(te)


class TestModel(unittest.TestCase):
    def test_learns_planted_signal_and_not_noise(self):
        _, te, p = fit(1.0)
        self.assertGreater(roc_auc_score((te["y_class"] == 2).astype(int), p["probability_bullish"]), 0.65)
        _, te, p = fit(0.0)
        self.assertLess(abs(roc_auc_score((te["y_class"] == 2).astype(int), p["probability_bullish"]) - 0.5), 0.03)

    def test_outputs(self):
        _, _, p = fit(1.0)
        for c in ("probability_bullish", "probability_neutral", "probability_bearish", "expected_return_5d", "prob_hit_tp",
                  "base_rate_bullish", "prob_dispersion"):
            self.assertIn(c, p)

    def test_storage_versioning_promote_reject(self):
        cfg = make_cfg()
        ctx = make_ctx(cfg)
        m, _, _ = fit(1.0, cfg)
        for cls in (DatabaseModelStorage, LocalModelStorage):
            with self.subTest(backend=cls.backend):
                s = cls(cfg, ctx.repo)
                v1 = s.save(m, {"classifiers": ["logistic"], "feature_version": "f1", "metrics": {"auc_mean": 0.6}})
                s.promote(v1)
                v2 = s.save(m, {"classifiers": ["logistic"], "feature_version": "f1"})
                s.reject(v2, "lebih buruk")
                loaded, active = s.load_active_model()
                self.assertEqual(active["version"], v1)                  # model lama tetap aktif
                self.assertEqual(loaded.meta["model_version"], v1)
                v3 = s.save(m, {"classifiers": ["logistic"], "feature_version": "f1"})
                s.promote(v3)
                st = s.list_versions().set_index("version")["status"]
                self.assertEqual((st[v1], st[v2], st[v3]), ("RETIRED", "REJECTED", "ACTIVE"))
                self.assertEqual(int(v3.split("_v")[1]), int(v1.split("_v")[1]) + 2)   # tidak ada overwrite
        ctx.db.execute("UPDATE model_versions SET artifact_sha256 = 'x' WHERE status = 'ACTIVE'")
        import shutil
        shutil.rmtree(__import__("app.config", fromlist=["x"]).cache_dir(cfg) / "models", ignore_errors=True)
        with self.assertRaises(ModelStorageError):
            DatabaseModelStorage(cfg, ctx.repo).load_active_model()

    def test_promotion_gates(self):
        cfg = make_cfg()
        good = {"auc_mean": 0.56, "auc_std": 0.02, "ece_bullish": 0.03, "positive_fold_ratio": 0.75}
        self.assertEqual(evaluate_gates(good, cfg), [])
        self.assertEqual(len(evaluate_gates({**good, "auc_mean": 0.50, "auc_std": 0.09}, cfg)), 2)


if __name__ == "__main__":
    unittest.main()
