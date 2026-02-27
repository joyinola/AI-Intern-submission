"""
Check Me — Assessment Router
==============================
POST /assess  — Full breast cancer risk assessment for a single patient.
GET  /example — Pre-built example patients covering all three risk tiers.
"""

from fastapi import APIRouter, HTTPException, Depends

from api.schemas import PatientInput, AssessmentResponse, FlagOut, FeatureOut
from api.dependencies import get_model
from model.core.decision import make_decision

router = APIRouter(tags=["Assessment"])


@router.post("/assess", response_model=AssessmentResponse)
def assess_patient(patient: PatientInput, model=Depends(get_model)):
    """
    Run a full breast cancer risk assessment.

    Returns risk tier (GREEN / YELLOW / RED), ML probability from
    GradientBoosting model, triggered clinical safety flags,
    feature-level explanations, and personalised clinical recommendations.
    """
    pipeline, metrics = model
    patient_dict = patient.model_dump(exclude={"patient_ref", "clinician_ref"})

    try:
        decision = make_decision(
            patient=patient_dict,
            pipeline=pipeline,
            feature_importances=dict(metrics.get("feature_importance") or {k:v for k,v in metrics.get("top_features",[])}  ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    return AssessmentResponse(
        risk_tier=decision.risk_tier,
        risk_score=decision.risk_score,
        clinical_risk_score=decision.clinical_risk_score,
        confidence=decision.confidence,
        tier_color=decision.tier_color,
        tier_label=decision.tier_label,
        tier_sub=decision.tier_sub,
        clinical_flags=[
            FlagOut(
                id=f.id,
                condition=f.condition,
                rationale=f.rationale,
                urgency=f.urgency,
            )
            for f in decision.clinical_flags
        ],
        top_features=[FeatureOut(**f) for f in decision.top_features],
        recommendations=decision.recommendations,
        next_steps=decision.next_steps,
        disclaimer=decision.disclaimer,
    )


@router.get("/example", tags=["Assessment"])
def example_patients():
    """Three pre-built patients covering all three risk tiers (GREEN/YELLOW/RED)."""
    return {
        "green_low_risk": {
            "age": 42, "bmi": 24.5, "alcohol_drinks_week": 2,
            "mean_radius": 12.3, "mean_texture": 17.8, "mean_perimeter": 78.9,
            "mean_area": 476, "mean_smoothness": 0.091, "mean_compactness": 0.076,
            "mean_concavity": 0.041, "mean_concave_points": 0.022,
            "mean_symmetry": 0.174, "mean_fractal_dimension": 0.062,
            "worst_radius": 14.1, "worst_texture": 25.1, "worst_area": 572,
            "worst_concavity": 0.120, "worst_concave_points": 0.072,
            "family_history_bc": 0, "prior_biopsy": 0, "hrt_use": 0,
            "brca_mutation": 0, "dense_breast": 0, "palpable_lump": 0,
            "nipple_discharge": 0, "skin_changes": 0,
        },
        "yellow_moderate_risk": {
            "age": 55, "bmi": 29.0, "alcohol_drinks_week": 8,
            "mean_radius": 15.0, "mean_texture": 20.5, "mean_perimeter": 98.0,
            "mean_area": 720, "mean_smoothness": 0.100, "mean_compactness": 0.110,
            "mean_concavity": 0.095, "mean_concave_points": 0.055,
            "mean_symmetry": 0.185, "mean_fractal_dimension": 0.063,
            "worst_radius": 18.2, "worst_texture": 27.0, "worst_area": 1050,
            "worst_concavity": 0.320, "worst_concave_points": 0.140,
            "family_history_bc": 1, "prior_biopsy": 1, "hrt_use": 1,
            "brca_mutation": 0, "dense_breast": 1, "palpable_lump": 0,
            "nipple_discharge": 0, "skin_changes": 0,
        },
        "red_high_risk": {
            "age": 45, "bmi": 27.0, "alcohol_drinks_week": 5,
            "mean_radius": 19.5, "mean_texture": 24.0, "mean_perimeter": 128.0,
            "mean_area": 1200, "mean_smoothness": 0.110, "mean_compactness": 0.180,
            "mean_concavity": 0.210, "mean_concave_points": 0.110,
            "mean_symmetry": 0.200, "mean_fractal_dimension": 0.066,
            "worst_radius": 24.0, "worst_texture": 32.0, "worst_area": 2100,
            "worst_concavity": 0.75, "worst_concave_points": 0.22,
            "family_history_bc": 1, "prior_biopsy": 0, "hrt_use": 0,
            "brca_mutation": 1, "dense_breast": 1, "palpable_lump": 1,
            "nipple_discharge": 1, "skin_changes": 1,
        },
    }
