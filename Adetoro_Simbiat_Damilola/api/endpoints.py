
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from api.main import app
from api.schemas import PatientInput, AssessmentResponse, FlagOut, FeatureOut
from model import (
    FEATURE_COLS,
    FEATURE_DESCRIPTIONS,
    CLINICAL_FLAGS,
    load_model,
    make_decision,
)


pipeline, metrics = load_model("model/")


@app.get("/model-info")
def model_info():
    return {
        "system": "Check Me — Breast Cancer Risk Stratification",
        "model_type": "Logistic Regression + Clinical Safety Flags",
        "dataset": {
            "source": "Synthetic (based on UCI Wisconsin BC Diagnostic distributions + clinical risk factors)",
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
            "ELEVATED": "P 0.28–0.48 or non-urgent flags",
            "HIGH":     "P 0.48–0.72 or prompt flags",
            "URGENT":   "P ≥ 0.72 or any IMMEDIATE clinical flag",
        },
        "clinical_guidelines": ["ACS 2023", "NICE NG101", "BI-RADS 5th Edition", "WHO Breast Cancer 2023"],
        "feature_importance": metrics["feature_importance"],
    }


@app.get("/clinical-flags")
def clinical_flags():
    return {
        "flags": [
            {"id": f.id, "condition": f.condition, "rationale": f.rationale, "urgency": f.urgency}
            for f in CLINICAL_FLAGS
        ],
        "note": (
            "Flags are evaluated independently of the ML model. "
            "IMMEDIATE flags escalate to URGENT tier regardless of ML score. "
            "PROMPT flags escalate to HIGH tier or above."
        ),
    }


@app.get("/example")
def example_patients():
    return {
        "routine_screening": {
            "age": 42, "bmi": 24.5, "alcohol_drinks_week": 2,
            "radius_mean": 12.3, "texture_mean": 17.8, "perimeter_mean": 78.9,
            "area_mean": 476, "smoothness_mean": 0.091, "compactness_mean": 0.076,
            "concavity_mean": 0.041, "concave_points_mean": 0.022,
            "symmetry_mean": 0.174, "fractal_dim_mean": 0.062,
            "radius_worst": 14.1, "texture_worst": 25.1, "area_worst": 572,
            "concavity_worst": 0.120, "concave_pts_worst": 0.072,
            "family_history_bc": 0, "prior_biopsy": 0, "hrt_use": 0,
            "brca_mutation": 0, "dense_breast": 0, "palpable_lump": 0,
            "nipple_discharge": 0, "skin_changes": 0,
        },
        "high_risk": {
            "age": 58, "bmi": 30.1, "alcohol_drinks_week": 10,
            "radius_mean": 17.2, "texture_mean": 22.1, "perimeter_mean": 114.8,
            "area_mean": 932, "smoothness_mean": 0.108, "compactness_mean": 0.151,
            "concavity_mean": 0.168, "concave_points_mean": 0.091,
            "symmetry_mean": 0.194, "fractal_dim_mean": 0.064,
            "radius_worst": 21.4, "texture_worst": 30.1, "area_worst": 1438,
            "concavity_worst": 0.468, "concave_pts_worst": 0.187,
            "family_history_bc": 1, "prior_biopsy": 1, "hrt_use": 1,
            "brca_mutation": 0, "dense_breast": 1, "palpable_lump": 1,
            "nipple_discharge": 0, "skin_changes": 0,
        },
        "urgent_flags": {
            "age": 45, "bmi": 27.0, "alcohol_drinks_week": 5,
            "radius_mean": 19.5, "texture_mean": 24.0, "perimeter_mean": 128,
            "area_mean": 1200, "smoothness_mean": 0.110, "compactness_mean": 0.180,
            "concavity_mean": 0.210, "concave_points_mean": 0.110,
            "symmetry_mean": 0.200, "fractal_dim_mean": 0.066,
            "radius_worst": 24.0, "texture_worst": 32.0, "area_worst": 2100,
            "concavity_worst": 0.75, "concave_pts_worst": 0.22,
            "family_history_bc": 1, "prior_biopsy": 0, "hrt_use": 0,
            "brca_mutation": 1, "dense_breast": 1, "palpable_lump": 1,
            "nipple_discharge": 1, "skin_changes": 1,
        },
    }


@app.post("/assess", response_model=AssessmentResponse)
def assess_patient(patient: PatientInput):
    """
    Run a full breast cancer risk assessment for a patient.

    Returns risk tier (ROUTINE/ELEVATED/HIGH/URGENT), ML probability score,
    triggered clinical flags, feature-level explanations, and clinical recommendations.
    """
    patient_dict = patient.model_dump(exclude={"patient_ref", "clinician_ref"})

    try:
        decision = make_decision(
            patient=patient_dict,
            pipeline=pipeline,
            feature_importances=metrics["feature_importance"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    return AssessmentResponse(
        risk_tier=decision.risk_tier,
        risk_score=decision.risk_score,
        confidence=decision.confidence,
        clinical_flags=[
            FlagOut(id=f.id, condition=f.condition, rationale=f.rationale, urgency=f.urgency)
            for f in decision.clinical_flags
        ],
        top_features=[FeatureOut(**f) for f in decision.top_features],
        recommendations=decision.recommendations,
        next_steps=decision.next_steps,
        disclaimer=decision.disclaimer,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
