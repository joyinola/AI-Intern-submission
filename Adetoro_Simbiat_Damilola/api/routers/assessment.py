"""
Check Me — Assessment Router
==============================
POST /assess  — Full breast cancer risk assessment for a single patient.
GET  /example — Pre-built example patients for testing all four risk tiers.
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

    Returns risk tier (ROUTINE / ELEVATED / HIGH / URGENT), ML probability,
    triggered clinical safety flags, feature-level explanations, and
    personalised clinical recommendations.
    """
    pipeline, metrics = model
    patient_dict = patient.model_dump(exclude={"patient_ref", "clinician_ref"})

    try:
        decision = make_decision(
            patient=patient_dict,
            pipeline=pipeline,
            feature_importances=metrics["feature_importance"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")

    return AssessmentResponse(
        risk_tier=decision.risk_tier,
        risk_score=decision.risk_score,
        confidence=decision.confidence,
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
    """Three pre-built patients covering routine / high / urgent risk tiers."""
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
