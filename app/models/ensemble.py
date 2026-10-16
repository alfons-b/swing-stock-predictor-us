"""SwingModel: gabungan classifier (terkalibrasi, ensemble), regressor, dan model hit-TP.

Output per baris:
probability_bearish/neutral/bullish, prob_dispersion (uncertainty antar anggota ensemble),
expected_return_{h}d, expected_mfe, expected_mae, prob_hit_tp, prob_hit_sl_first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd


@dataclass
class SwingModel:
    features: list
    horizon: int
    horizons: list
    classifiers: dict = field(default_factory=dict)       # nama → estimator terkalibrasi
    regressors: dict = field(default_factory=dict)        # target → estimator
    hit_tp_model: object = None
    hit_sl_model: object = None
    meta: dict = field(default_factory=dict)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(f"Fitur hilang saat prediksi: {missing[:5]}...")
        # juga melindungi model yang sudah tersimpan dari nilai non-finite / ekstrem
        return df[self.features].astype(float).replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df)
        probs = []
        for est in self.classifiers.values():
            p = est.predict_proba(X)
            classes = list(getattr(est, "classes_", [0, 1, 2]))
            full = np.zeros((len(X), 3))
            for j, cl in enumerate(classes):
                full[:, int(cl)] = p[:, j]
            probs.append(full)
        P = np.mean(probs, axis=0)
        out = pd.DataFrame(index=df.index)
        out["probability_bearish"], out["probability_neutral"], out["probability_bullish"] = P[:, 0], P[:, 1], P[:, 2]
        out["prob_dispersion"] = np.std([p[:, 2] for p in probs], axis=0) * (2 if len(probs) == 2 else 1) if len(probs) > 1 else 0.0
        out["base_rate_bullish"] = self.meta.get("base_rate_bullish", np.nan)
        for target, est in self.regressors.items():
            out[f"expected_{target}"] = est.predict(X)
        if self.hit_tp_model is not None:
            out["prob_hit_tp"] = self.hit_tp_model.predict_proba(X)[:, 1]
        if self.hit_sl_model is not None:
            out["prob_hit_sl_first"] = self.hit_sl_model.predict_proba(X)[:, 1]
        return out

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "SwingModel":
        return joblib.load(path)
