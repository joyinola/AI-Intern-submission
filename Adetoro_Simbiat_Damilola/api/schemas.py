"""
Check Me — API Schemas
========================
Pydantic request/response models for all endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    # FNA nucleus measurements
    radius_mean:         float = Field(..., ge=4,    le=35,   description="Mean nucleus radius")
    texture_mean:        float = Field(..., ge=5,    le=50,   description="Mean nucleus texture (grey-scale SD)")
    perimeter_mean:      float = Field(..., ge=30,   le=200,  description="Mean nucleus perimeter")
    area_mean:           float = Field(..., ge=100,  le=2500, description="Mean nucleus area (µm²)")
    smoothness_mean:     float = Field(..., ge=0.05, le=0.17, description="Mean nucleus smoothness")
    compactness_mean:    float = Field(..., ge=0.01, le=0.35, description="Mean nucleus compactness")
    concavity_mean:      float = Field(..., ge=0.0,  le=0.45, description="Mean nucleus concavity")
    concave_points_mean: float = Field(..., ge=0.0,  le=0.20, description="Mean concave nucleus points")
    symmetry_mean:       float = Field(..., ge=0.1,  le=0.3,  description="Mean nucleus symmetry")
    fractal_dim_mean:    float = Field(..., ge=0.04, le=0.10, description="Mean fractal dimension")
    radius_worst:        float = Field(..., ge=6,    le=36,   description="Worst-case nucleus radius")
    texture_worst:       float = Field(..., ge=10,   le=55,   description="Worst-case nucleus texture")
    area_worst:          float = Field(..., ge=150,  le=4000, description="Worst-case nucleus area (µm²)")
    concavity_worst:     float = Field(..., ge=0.0,  le=1.0,  description="Worst-case nucleus concavity")
    concave_pts_worst:   float = Field(..., ge=0.0,  le=0.30, description="Worst-case concave points")
    # Clinical factors
    age:                 float = Field(..., ge=18,   le=100,  description="Patient age")
    bmi:                 float = Field(..., ge=14,   le=60,   description="BMI (kg/m²)")
    alcohol_drinks_week: float = Field(0,   ge=0,    le=30,   description="Alcohol units per week")
    family_history_bc:   int   = Field(0,   ge=0,    le=1,    description="First-degree family history of BC")
    prior_biopsy:        int   = Field(0,   ge=0,    le=1,    description="Prior breast biopsy")
    hrt_use:             int   = Field(0,   ge=0,    le=1,    description="Hormone replacement therapy use")
    brca_mutation:       int   = Field(0,   ge=0,    le=1,    description="Known BRCA1/2 mutation")
    dense_breast:        int   = Field(0,   ge=0,    le=1,    description="Dense breast tissue (BI-RADS C/D)")
    palpable_lump:       int   = Field(0,   ge=0,    le=1,    description="Palpable breast lump")
    nipple_discharge:    int   = Field(0,   ge=0,    le=1,    description="Spontaneous nipple discharge")
    skin_changes:        int   = Field(0,   ge=0,    le=1,    description="Skin dimpling / peau d'orange")
    # Audit metadata (not used in model)
    patient_ref:         Optional[str] = Field(None, description="De-identified patient reference")
    clinician_ref:       Optional[str] = Field(None, description="Requesting clinician ID")


class FlagOut(BaseModel):
    id: str
    condition: str
    rationale: str
    urgency: str


class FeatureOut(BaseModel):
    feature: str
    description: str
    value: float
    importance: float


class AssessmentResponse(BaseModel):
    risk_tier: str
    risk_score: float
    confidence: str
    clinical_flags: list[FlagOut]
    top_features: list[FeatureOut]
    recommendations: list[str]
    next_steps: list[str]
    disclaimer: str
