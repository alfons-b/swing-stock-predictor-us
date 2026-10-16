"""Probability calibration dengan split berbasis waktu (bukan acak).

Data training dibagi: [awal ... cal_start - embargo] → fit model; [cal_start ... akhir] → kalibrasi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def time_calibration_split(dates: pd.Series, fraction: float, embargo: int):
    ud = np.sort(dates.unique())
    if fraction <= 0 or len(ud) < 100:
        return np.ones(len(dates), bool), np.zeros(len(dates), bool)
    cut = int(len(ud) * (1 - fraction))
    fit_end = ud[max(0, cut - embargo - 1)]
    cal_start = ud[cut]
    return (dates <= fit_end).to_numpy(), (dates >= cal_start).to_numpy()


def fit_calibrated(estimator, X: pd.DataFrame, y: np.ndarray, dates: pd.Series, method: str,
                   fraction: float, embargo: int):
    if method in (None, "none"):
        estimator.fit(X, y)
        return estimator
    fit_m, cal_m = time_calibration_split(dates, fraction, embargo)
    if cal_m.sum() < 500 or len(np.unique(y[cal_m])) < len(np.unique(y)):
        estimator.fit(X, y)
        return estimator
    estimator.fit(X[fit_m], y[fit_m])
    cal = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    cal.fit(X[cal_m], y[cal_m])
    return cal
