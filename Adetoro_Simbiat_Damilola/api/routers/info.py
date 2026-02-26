"""
Check Me — Info Router
========================
GET /model-info       — Model metadata, performance metrics, and feature importances.
GET /clinical-flags   — All 8 safety flags with conditions, rationale, and urgency.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_model
from model.core.flags import CLINICAL_FLAGS

router = APIRouter(tags=["Information"])


@router.get("/model-info")
def model_info(model=Depends(get_model)):
    """Return model metadata, CV performance, risk tier definitions, and feature importance."""
    _, metrics = model
    return {
        "system": "Check Me — Breast Cancer Risk Stratification",
        "model_type": "Logistic Regression + Clinical Safety Flags",
        "dataset": {
            "source": "Synthetic (UCI Wisconsin BC Diagnostic distributions + clinical risk factors)",
            "n_samples": metrics["n_samples"],
            "malignant_rate": f"{metrics['malignant_rate']:.1%}",
        },
        "performance": {
            "roc_auc":    f"{metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}",
            "f1":         f"{metrics['cv_f1_mean']:.3f}",
            "recall":     f"{metrics['cv_recall_mean']:.3f}  (sensitivity — minimising missed malignancies)",
            "precision":  f"{metrics['cv_precision_mean']:.3f}",
            "brier_score": metrics["brier_score"],
        },
        "risk_tiers": {
            "ROUTINE":  "P < 0.28 and no urgent flags",
            "ELEVATED": "P 0.28–0.48",
            "HIGH":     "P 0.48–0.72 or PROMPT flags",
            "URGENT":   "P ≥ 0.72 or any IMMEDIATE clinical flag",
        },
        "clinical_guidelines": [
            "ACS 2023",
            "NICE NG101",
            "BI-RADS 5th Edition",
            "WHO Breast Cancer 2023",
        ],
        "feature_importance": metrics["feature_importance"],
    }


@router.get("/clinical-flags")
def clinical_flags():
    """Return all 8 safety flags with conditions, rationale, and urgency levels."""
    return {
        "flags": [
            {
                "id":        f.id,
                "condition": f.condition,
                "rationale": f.rationale,
                "urgency":   f.urgency,
            }
            for f in CLINICAL_FLAGS
        ],
        "note": (
            "Flags are evaluated independently of the ML model. "
            "IMMEDIATE flags escalate to URGENT tier regardless of ML score. "
            "PROMPT flags escalate to HIGH tier or above."
        ),
    }
