"""
Check Me — Breast Cancer Risk Model
=====================================
Hybrid Architecture:
  1. Hard clinical safety flags — always surfaced regardless of ML output
  2. Logistic Regression — trained on Wisconsin-style synthetic dataset
  3. Permutation importance — interpretability without external libraries
  4. Four-tier risk output: ROUTINE / ELEVATED / HIGH / URGENT
     with personalised recommendations at each level

Clinical references:
  - American Cancer Society guidelines (2023)
  - BI-RADS lexicon (ACR 5th edition)
  - NICE NG101 Familial Breast Cancer
  - WHO Breast Cancer Fact Sheet 2023
"""

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")



# 1.  CLINICAL SAFETY FLAGS
#     These are always evaluated and always surfaced to the clinician.
#     They do NOT forcibly override the model but escalate to URGENT tier.


@dataclass
class ClinicalFlag:
    id: str
    condition: str
    rationale: str
    urgency: str   # "IMMEDIATE" | "PROMPT" | "MONITOR"


CLINICAL_FLAGS: list[ClinicalFlag] = [
    ClinicalFlag("brca_palpable",    "Known BRCA mutation + palpable lump",       "ACS: immediate surgical referral recommended",              "IMMEDIATE"),
    ClinicalFlag("skin_nipple",      "Skin changes + nipple discharge",           "BI-RADS: combined signs associated with inflammatory BC",   "IMMEDIATE"),
    ClinicalFlag("brca_known",       "Known BRCA1/2 pathogenic variant",          "NICE NG101: high-risk surveillance pathway",               "PROMPT"),
    ClinicalFlag("fh_lump",          "Strong family history + palpable lump",     "ACS: combination warrants urgent imaging referral",        "PROMPT"),
    ClinicalFlag("skin_change",      "Skin dimpling or peau d'orange",            "Classic sign of underlying malignancy",                    "IMMEDIATE"),
    ClinicalFlag("high_concavity",   "Concavity worst > 0.7 (FNA)",              "High nuclear concavity — cytological malignancy marker",   "PROMPT"),
    ClinicalFlag("large_area",       "Nuclear area worst > 2000 µm²",            "Markedly enlarged nuclei — strong malignancy predictor",   "PROMPT"),
    ClinicalFlag("age_brca",         "Age > 30 with BRCA mutation",              "NICE: annual MRI from age 30 for BRCA carriers",           "PROMPT"),
]


def evaluate_clinical_flags(row: dict) -> list[ClinicalFlag]:
    """Evaluate and return triggered clinical flags for a patient."""
    triggered = []

    if row.get("brca_mutation") and row.get("palpable_lump"):
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "brca_palpable"))
    if row.get("skin_changes") and row.get("nipple_discharge"):
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "skin_nipple"))
    if row.get("brca_mutation"):
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "brca_known"))
    if row.get("family_history_bc") and row.get("palpable_lump"):
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "fh_lump"))
    if row.get("skin_changes"):
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "skin_change"))
    if row.get("concavity_worst", 0) > 0.70:
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "high_concavity"))
    if row.get("area_worst", 0) > 2000:
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "large_area"))
    if row.get("brca_mutation") and row.get("age", 0) > 30:
        triggered.append(next(f for f in CLINICAL_FLAGS if f.id == "age_brca"))

    # Deduplicate by id
    seen = set()
    unique = []
    for f in triggered:
        if f.id not in seen:
            seen.add(f.id)
            unique.append(f)
    return unique



# 2.  MODEL TRAINING


FEATURE_COLS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dim_mean",
    "radius_worst", "texture_worst", "area_worst",
    "concavity_worst", "concave_pts_worst",
    "age", "bmi", "alcohol_drinks_week",
    "family_history_bc", "prior_biopsy", "hrt_use",
    "brca_mutation", "dense_breast",
    "palpable_lump", "nipple_discharge", "skin_changes",
]

FEATURE_DESCRIPTIONS = {
    "radius_mean":          "Mean cell nucleus radius (FNA)",
    "texture_mean":         "Mean nucleus texture (grey-scale SD)",
    "perimeter_mean":       "Mean nucleus perimeter",
    "area_mean":            "Mean nucleus area (µm²)",
    "smoothness_mean":      "Nucleus contour smoothness",
    "compactness_mean":     "Nucleus compactness (perimeter²/area)",
    "concavity_mean":       "Severity of concave nucleus portions",
    "concave_points_mean":  "Number of concave nucleus points",
    "symmetry_mean":        "Nucleus symmetry",
    "fractal_dim_mean":     "Fractal dimension (coastline approx.)",
    "radius_worst":         "Largest nucleus radius (worst case)",
    "texture_worst":        "Worst-case nucleus texture",
    "area_worst":           "Largest nucleus area (worst case, µm²)",
    "concavity_worst":      "Worst-case nucleus concavity",
    "concave_pts_worst":    "Worst-case concave nucleus points",
    "age":                  "Patient age (years)",
    "bmi":                  "Body mass index (kg/m²)",
    "alcohol_drinks_week":  "Alcohol units per week",
    "family_history_bc":    "First-degree relative with breast cancer",
    "prior_biopsy":         "Previous breast biopsy",
    "hrt_use":              "Current hormone replacement therapy",
    "brca_mutation":        "Known BRCA1/2 pathogenic mutation",
    "dense_breast":         "Dense breast tissue (BI-RADS C/D)",
    "palpable_lump":        "Palpable breast lump on exam",
    "nipple_discharge":     "Spontaneous nipple discharge",
    "skin_changes":         "Skin dimpling / peau d'orange",
}


def train_model(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS]
    y = df["malignant"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.8,
            class_weight="balanced",   # penalise missed malignancies
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
        )),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc     = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
    cv_f1      = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    cv_recall  = cross_val_score(pipeline, X, y, cv=skf, scoring="recall")
    cv_prec    = cross_val_score(pipeline, X, y, cv=skf, scoring="precision")

    pipeline.fit(X, y)

    proba  = pipeline.predict_proba(X)[:, 1]
    brier  = brier_score_loss(y, proba)
    importances = _permutation_importance(pipeline, X, y)

    metrics = {
        "cv_roc_auc_mean":    float(cv_roc.mean()),
        "cv_roc_auc_std":     float(cv_roc.std()),
        "cv_f1_mean":         float(cv_f1.mean()),
        "cv_recall_mean":     float(cv_recall.mean()),
        "cv_precision_mean":  float(cv_prec.mean()),
        "brier_score":        float(brier),
        "feature_importance": importances,
        "n_samples":          len(df),
        "malignant_rate":     float(y.mean()),
    }
    return {"pipeline": pipeline, "metrics": metrics}


def _permutation_importance(pipeline, X, y, n_repeats=8):
    baseline = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    rng = np.random.default_rng(42)
    importances = {}
    for col in FEATURE_COLS:
        scores = []
        for _ in range(n_repeats):
            X_p = X.copy()
            X_p[col] = rng.permutation(X_p[col].values)
            scores.append(baseline - roc_auc_score(y, pipeline.predict_proba(X_p)[:, 1]))
        importances[col] = float(np.mean(scores))
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))



# 3.  RECOMMENDATION LOGIC


@dataclass
class CheckMeDecision:
    risk_tier: str           # ROUTINE | ELEVATED | HIGH | URGENT
    risk_score: float        # 0.0–1.0 ML probability
    confidence: str          # HIGH | MODERATE | LOW
    clinical_flags: list     # triggered ClinicalFlag objects
    top_features: list[dict]
    recommendations: list[str]
    next_steps: list[str]
    disclaimer: str
    raw_input: dict = field(default_factory=dict)


TIER_RECOMMENDATIONS = {
    "ROUTINE": {
        "recommendations": [
            "Continue annual mammography screening per ACS age-appropriate guidelines",
            "Monthly self-breast examination recommended",
            "Maintain healthy weight and limit alcohol to < 7 units/week",
            "Next scheduled mammogram as per standard programme",
        ],
        "next_steps": [
            "Routine screening mammogram (annual if ≥ 40)",
            "Clinical breast exam at next GP visit",
            "Reassess risk score if new symptoms develop",
        ],
    },
    "ELEVATED": {
        "recommendations": [
            "Earlier or supplemental screening discussion with GP/oncologist",
            "Consider breast ultrasound as adjunct to mammography",
            "Document and monitor all risk factors at follow-up visits",
            "Discuss lifestyle modifications: weight management, alcohol reduction",
        ],
        "next_steps": [
            "GP referral for clinical breast assessment within 6 weeks",
            "Discuss supplemental MRI if dense breast tissue confirmed",
            "Genetic counselling referral if family history suggests hereditary risk",
        ],
    },
    "HIGH": {
        "recommendations": [
            "Prompt specialist referral to breast clinic or oncology unit",
            "Annual breast MRI alongside mammography (NICE NG101)",
            "Discussion of risk-reduction options (chemoprevention, lifestyle)",
            "Formal genetic risk assessment recommended",
        ],
        "next_steps": [
            "Urgent breast clinic referral within 2 weeks",
            "Diagnostic mammogram + ultrasound evaluation",
            "Genetic counselling appointment if BRCA/family risk",
            "Oncology nurse specialist involvement",
        ],
    },
    "URGENT": {
        "recommendations": [
            "⚠️  Immediate specialist referral — do not delay",
            "Two-week-wait (2WW) urgent cancer pathway referral (UK) / urgent oncology consult",
            "Physical examination findings require prompt imaging workup",
            "Do not reassure patient without complete imaging and clinical evaluation",
        ],
        "next_steps": [
            "2WW urgent breast clinic referral TODAY",
            "Triple assessment: clinical exam + mammogram + FNA/core biopsy",
            "Multidisciplinary team (MDT) review of imaging and pathology",
            "Patient counselling and psychological support resources",
        ],
    },
}


def make_decision(
    patient: dict,
    pipeline,
    feature_importances: dict,
) -> CheckMeDecision:

    # ML probability
    X_in = pd.DataFrame([{col: patient.get(col, 0) for col in FEATURE_COLS}])
    risk_prob = float(pipeline.predict_proba(X_in)[0, 1])

    # Clinical flags
    flags = evaluate_clinical_flags(patient)
    has_immediate = any(f.urgency == "IMMEDIATE" for f in flags)
    has_prompt    = any(f.urgency == "PROMPT" for f in flags)

    # Risk tier
    if has_immediate or risk_prob >= 0.72:
        tier = "URGENT"
    elif has_prompt or risk_prob >= 0.48:
        tier = "HIGH"
    elif risk_prob >= 0.28:
        tier = "ELEVATED"
    else:
        tier = "ROUTINE"

    # Confidence
    d = abs(risk_prob - 0.5)
    confidence = "HIGH" if (d > 0.30 or has_immediate) else "MODERATE" if d > 0.12 else "LOW"

    # Top contributing features
    top_features = []
    for feat, imp in list(feature_importances.items())[:6]:
        top_features.append({
            "feature": feat,
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            "value": round(patient.get(feat, 0), 4),
            "importance": round(imp, 5),
        })

    recs = TIER_RECOMMENDATIONS[tier]

    return CheckMeDecision(
        risk_tier=tier,
        risk_score=round(risk_prob, 4),
        confidence=confidence,
        clinical_flags=flags,
        top_features=top_features,
        recommendations=recs["recommendations"],
        next_steps=recs["next_steps"],
        disclaimer=(
            "Check Me is a clinical decision SUPPORT tool. "
            "It does not diagnose breast cancer. "
            "All outputs require validation by a qualified clinician. "
            "Do not use as a sole basis for clinical action. "
            "Based on: ACS 2023 guidelines, NICE NG101, BI-RADS 5th ed."
        ),
        raw_input=patient,
    )



# 4.  PERSISTENCE


def save_model(pipeline, metrics, path="model/"):
    Path(path).mkdir(exist_ok=True)
    with open(f"{path}/pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(f"{path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_model(path="model/"):
    with open(f"{path}/pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(f"{path}/metrics.json") as f:
        metrics = json.load(f)
    return pipeline, metrics



# 5.  ENTRY POINT


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset import generate_breast_cancer_dataset

    Path("data").mkdir(exist_ok=True)
    print("Generating synthetic breast cancer dataset...")
    df = generate_breast_cancer_dataset(n=3000)
    df.to_csv("data/breast_cancer_synthetic.csv", index=False)
    print(f"Dataset: {len(df)} rows | Malignant rate: {df.malignant.mean():.1%}\n")

    print("Training Check Me model...")
    result  = train_model(df)
    pipeline = result["pipeline"]
    metrics  = result["metrics"]

    print(f"\n{'='*55}")
    print("CHECK ME — MODEL EVALUATION (5-Fold Stratified CV)")
    print(f"{'='*55}")
    print(f"  ROC-AUC   : {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    print(f"  F1        : {metrics['cv_f1_mean']:.3f}")
    print(f"  Recall    : {metrics['cv_recall_mean']:.3f}  ← minimising missed malignancies")
    print(f"  Precision : {metrics['cv_precision_mean']:.3f}")
    print(f"  Brier     : {metrics['brier_score']:.4f}  (calibration; lower = better)")

    print(f"\n{'='*55}")
    print("TOP FEATURE IMPORTANCES (Permutation AUC Drop)")
    print(f"{'='*55}")
    for feat, imp in list(metrics["feature_importance"].items())[:10]:
        bar = "█" * max(1, int(imp * 600))
        print(f"  {feat:<28}  {imp:+.5f}  {bar}")

    save_model(pipeline, metrics)
    print("\n✓ Model saved to model/")
    print("  Run:  uvicorn api:app --reload --port 8000")
