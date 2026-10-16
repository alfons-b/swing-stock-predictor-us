"""Registry model. Library opsional di-import lazy; bila tidak terpasang, kandidat dilewati.

Keputusan desain: baseline sederhana (Logistic Regression) selalu ikut dibandingkan.
Model kompleks hanya dipilih bila menang secara walk-forward, bukan karena populer.
"""
from __future__ import annotations

import importlib.util

from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.utils.logging_utils import get_logger

log = get_logger(__name__)

_OPTIONAL = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}


def is_available(name: str) -> bool:
    mod = _OPTIONAL.get(name)
    return mod is None or importlib.util.find_spec(mod) is not None


def available_models(names: list[str]) -> list[str]:
    out = []
    for n in names:
        if is_available(n):
            out.append(n)
        else:
            log.warning("Model %s dilewati: library belum terpasang (pip install %s)", n, _OPTIONAL[n])
    return out


def make_classifier(name: str, seed: int = 42):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=1000))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=200,
                                                    max_samples=0.3, max_features="sqrt", n_jobs=-1, random_state=seed))
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=300,
                                              l2_regularization=1.0, early_stopping=False, random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                             colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", random_state=seed,
                             objective="multi:softprob", eval_metric="mlogloss")
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=3, random_seed=seed,
                                  loss_function="MultiClass", verbose=False)
    raise ValueError(f"Classifier tidak dikenal: {name}")


def make_binary_classifier(family: str, seed: int = 42):
    if family == "logistic":
        return make_classifier("logistic", seed)
    if family == "lightgbm" and is_available("lightgbm"):
        return make_classifier("lightgbm", seed)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=300,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)


def make_regressor(name: str, seed: int = 42):
    if name in ("ridge", "logistic"):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=200, max_samples=0.3,
                                                   max_features="sqrt", n_jobs=-1, random_state=seed))
    if name == "lightgbm" and is_available("lightgbm"):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.7, objective="huber", random_state=seed, verbose=-1)
    # default robust: HGB dengan loss absolute_error tahan outlier return
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=seed)


def regressor_family_for(classifier_name: str) -> str:
    return {"logistic": "ridge", "random_forest": "random_forest", "lightgbm": "lightgbm"}.get(classifier_name, "hist_gbm")
