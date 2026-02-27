"""
Check Me — Model Trainer (GradientBoosting / XGBoost-equivalent)
================================================================
Trains a Gradient Boosting Classifier on the augmented UCI breast cancer
dataset. GradientBoostingClassifier is sklearn's native XGBoost-equivalent
(same histogram-based boosting algorithm).

To swap to true XGBoost, replace:
    from sklearn.ensemble import GradientBoostingClassifier
with:
    from xgboost import XGBClassifier  (after: pip install xgboost)
and rename clf__sample_weight → clf__sample_weight (same param name).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

warnings.filterwarnings("ignore")


def train_model(df: pd.DataFrame) -> dict:
    """
    Train the Check Me GradientBoosting risk model.

    Args:
        df: DataFrame containing FEATURE_COLS + 'malignant' target column.

    Returns:
        dict with keys 'pipeline' and 'metrics'.
    """
    X = df[FEATURE_COLS]
    y = df["malignant"]

    # Up-weight malignant cases — missing a malignancy (FN) is worse than over-referring (FP)
    n_benign    = (y == 0).sum()
    n_malignant = (y == 1).sum()
    weight_map  = {0: 1.0, 1: n_benign / n_malignant}
    sample_weights = y.map(weight_map).values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])

    # 3-fold stratified CV (sufficient for n=569, faster than 5-fold)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_roc    = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
    cv_f1     = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    cv_recall = cross_val_score(pipeline, X, y, cv=skf, scoring="recall")
    cv_prec   = cross_val_score(pipeline, X, y, cv=skf, scoring="precision")

    # Fit on full dataset with sample weights
    pipeline.fit(X, y, clf__sample_weight=sample_weights)

    proba       = pipeline.predict_proba(X)[:, 1]
    brier       = brier_score_loss(y, proba)
    importances = _permutation_importance(pipeline, X, y)

    metrics = {
        "cv_roc_auc_mean":    float(cv_roc.mean()),
        "cv_roc_auc_std":     float(cv_roc.std()),
        "cv_f1_mean":         float(cv_f1.mean()),
        "cv_recall_mean":     float(cv_recall.mean()),
        "cv_precision_mean":  float(cv_prec.mean()),
        "brier_score":        float(brier),
        "feature_importance": importances,
        "n_samples":          int(len(df)),
        "malignant_rate":     float(y.mean()),
        "model_type":         "GradientBoostingClassifier (XGBoost-equivalent)",
    }

    return {"pipeline": pipeline, "metrics": metrics}


def _permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 4,
) -> dict[str, float]:
    """
    Permutation-based feature importance as AUC drop.
    Model-agnostic — works identically whether backend is GBT or XGBoost.
    Higher value = feature is more important to predictions.
    """
    baseline = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    rng = np.random.default_rng(42)
    importances: dict[str, float] = {}

    for col in FEATURE_COLS:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = rng.permutation(X_permuted[col].values)
            scores.append(
                baseline - roc_auc_score(y, pipeline.predict_proba(X_permuted)[:, 1])
            )
        importances[col] = float(np.mean(scores))

    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
