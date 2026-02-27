
from typing import Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    # FNA nucleus measurements — real UCI column naming convention
    mean_radius:            float = Field(..., ge=4,    le=35,   description="Mean nucleus radius")
    mean_texture:           float = Field(..., ge=5,    le=50,   description="Mean nucleus texture")
    mean_perimeter:         float = Field(..., ge=30,   le=200,  description="Mean nucleus perimeter")
    mean_area:              float = Field(..., ge=100,  le=2500, description="Mean nucleus area (µm²)")
    mean_smoothness:        float = Field(..., ge=0.05, le=0.17, description="Mean smoothness")
    mean_compactness:       float = Field(..., ge=0.01, le=0.35, description="Mean compactness")
    mean_concavity:         float = Field(..., ge=0.0,  le=0.45, description="Mean concavity")
    mean_concave_points:    float = Field(..., ge=0.0,  le=0.20, description="Mean concave points")
    mean_symmetry:          float = Field(..., ge=0.1,  le=0.3,  description="Mean symmetry")
    mean_fractal_dimension: float = Field(..., ge=0.04, le=0.10, description="Mean fractal dimension")
    worst_radius:           float = Field(..., ge=6,    le=36,   description="Worst nucleus radius")
    worst_texture:          float = Field(..., ge=10,   le=55,   description="Worst nucleus texture")
    worst_area:             float = Field(..., ge=150,  le=4000, description="Worst nucleus area (µm²)")
    worst_concavity:        float = Field(..., ge=0.0,  le=1.0,  description="Worst concavity")
    worst_concave_points:   float = Field(..., ge=0.0,  le=0.30, description="Worst concave points")
    # Clinical factors
    age:                    float = Field(..., ge=18,   le=100,  description="Patient age")
    bmi:                    float = Field(..., ge=14,   le=60,   description="BMI (kg/m²)")
    alcohol_drinks_week:    float = Field(0,   ge=0,    le=30,   description="Alcohol units per week")
    family_history_bc:      int   = Field(0,   ge=0,    le=1,    description="First-degree family history")
    prior_biopsy:           int   = Field(0,   ge=0,    le=1,    description="Prior breast biopsy")
    hrt_use:                int   = Field(0,   ge=0,    le=1,    description="HRT use")
    brca_mutation:          int   = Field(0,   ge=0,    le=1,    description="Known BRCA1/2 mutation")
    dense_breast:           int   = Field(0,   ge=0,    le=1,    description="Dense breast tissue")
    palpable_lump:          int   = Field(0,   ge=0,    le=1,    description="Palpable breast lump")
    nipple_discharge:       int   = Field(0,   ge=0,    le=1,    description="Nipple discharge")
    skin_changes:           int   = Field(0,   ge=0,    le=1,    description="Skin changes")
    patient_ref:            Optional[str] = Field(None, description="De-identified patient ref")
    clinician_ref:          Optional[str] = Field(None, description="Clinician ID")


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
    risk_tier: str       # GREEN | YELLOW | RED
    risk_score: float
    clinical_risk_score: int
    confidence: str
    tier_color: str
    tier_label: str
    tier_sub: str
    clinical_flags: list[FlagOut]
    top_features: list[FeatureOut]
    recommendations: list[str]
    next_steps: list[str]
    disclaimer: str
