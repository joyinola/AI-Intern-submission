"""
Check Me — Model Trainer
==========================
Trains a Logistic Regression pipeline on the synthetic breast cancer dataset.
Uses class_weight='balanced' to heavily penalise missed malignancies (false negatives).
Evaluates with 5-fold stratified cross-validation and permutation feature importance.

Design decisions:
  - Logistic Regression: fully inspectable coefficients, no black box
  - StandardScaler: required for regularised logistic regression
  - class_weight='balanced': recall prioritised over precision in oncology context
  - Permutation importance: interpretability without SHAP dependency
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

warnings.filterwarnings("ignore")


def train_model(df: pd.DataFrame) -> dict:
    """
    Train the Check Me risk model and return the fitted pipeline + metrics.

    Args:
        df: DataFrame containing FEATURE_COLS + 'malignant' target column.

    Returns:
        dict with keys 'pipeline' and 'metrics'.
    """
    X = df[FEATURE_COLS]
    y = df["malignant"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.8,
            class_weight="balanced",  # penalise missed malignancies
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
        )),
    ])

    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc    = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
    cv_f1     = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    cv_recall = cross_val_score(pipeline, X, y, cv=skf, scoring="recall")
    cv_prec   = cross_val_score(pipeline, X, y, cv=skf, scoring="precision")

    # Fit on full dataset
    pipeline.fit(X, y)

    proba       = pipeline.predict_proba(X)[:, 1]
    brier       = brier_score_loss(y, proba)
    importances = _permutation_importance(pipeline, X, y)

    metrics = {
        "cv_roc_auc_mean":   float(cv_roc.mean()),
        "cv_roc_auc_std":    float(cv_roc.std()),
        "cv_f1_mean":        float(cv_f1.mean()),
        "cv_recall_mean":    float(cv_recall.mean()),
        "cv_precision_mean": float(cv_prec.mean()),
        "brier_score":       float(brier),
        "feature_importance": importances,
        "n_samples":         int(len(df)),
        "malignant_rate":    float(y.mean()),
    }

    return {"pipeline": pipeline, "metrics": metrics}


def _permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 8,
) -> dict[str, float]:
    """
    Compute permutation-based feature importance as AUC drop.
    Higher value = more important feature.
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
