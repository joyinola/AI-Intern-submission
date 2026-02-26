"""
Check Me — Decision & Recommendation Logic
============================================
Combines ML probability, clinical safety flags, and risk tier thresholds
into a structured CheckMeDecision with patient-specific recommendations.

Risk tier thresholds (grounded in BI-RADS / NICE NG101 referral criteria):
  URGENT:   P ≥ 0.72  OR any IMMEDIATE clinical flag
  HIGH:     P 0.48–0.72 OR any PROMPT flag
  ELEVATED: P 0.28–0.48
  ROUTINE:  P < 0.28, no flags
"""

from dataclasses import dataclass, field

import pandas as pd

from .features import FEATURE_COLS, FEATURE_DESCRIPTIONS
from .flags import ClinicalFlag, evaluate_clinical_flags


# ── Recommendation content per tier ──────────────────────────────────────────

_TIER_RECOMMENDATIONS: dict[str, dict] = {
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
            "Two-week-wait (2WW) urgent cancer pathway referral / urgent oncology consult",
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

_DISCLAIMER = (
    "Check Me is a clinical decision SUPPORT tool. "
    "It does not diagnose breast cancer. "
    "All outputs require validation by a qualified clinician. "
    "Do not use as a sole basis for clinical action. "
    "Based on: ACS 2023 guidelines, NICE NG101, BI-RADS 5th ed."
)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class CheckMeDecision:
    risk_tier: str            # ROUTINE | ELEVATED | HIGH | URGENT
    risk_score: float         # 0.0–1.0 ML probability
    confidence: str           # HIGH | MODERATE | LOW
    clinical_flags: list[ClinicalFlag]
    top_features: list[dict]
    recommendations: list[str]
    next_steps: list[str]
    disclaimer: str
    raw_input: dict = field(default_factory=dict)


# ── Core decision function ────────────────────────────────────────────────────

def make_decision(
    patient: dict,
    pipeline,
    feature_importances: dict,
) -> CheckMeDecision:
    """
    Run the full decision pipeline for one patient.

    Steps:
      1. Get ML probability from the trained pipeline.
      2. Evaluate clinical safety flags independently.
      3. Determine risk tier (flags can escalate, never de-escalate).
      4. Compute confidence from distance-from-boundary.
      5. Attach top-6 feature importances with patient values.
      6. Return structured CheckMeDecision.
    """
    # 1. ML probability
    X_in = pd.DataFrame([{col: patient.get(col, 0) for col in FEATURE_COLS}])
    risk_prob = float(pipeline.predict_proba(X_in)[0, 1])

    # 2. Clinical flags (independent of ML)
    flags = evaluate_clinical_flags(patient)
    has_immediate = any(f.urgency == "IMMEDIATE" for f in flags)
    has_prompt    = any(f.urgency == "PROMPT"    for f in flags)

    # 3. Risk tier — flags can only escalate
    if has_immediate or risk_prob >= 0.72:
        tier = "URGENT"
    elif has_prompt or risk_prob >= 0.48:
        tier = "HIGH"
    elif risk_prob >= 0.28:
        tier = "ELEVATED"
    else:
        tier = "ROUTINE"

    # 4. Confidence (distance from 0.5 decision boundary)
    distance = abs(risk_prob - 0.5)
    if distance > 0.30 or has_immediate:
        confidence = "HIGH"
    elif distance > 0.12:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # 5. Top contributing features
    top_features = [
        {
            "feature":     feat,
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            "value":       round(patient.get(feat, 0), 4),
            "importance":  round(imp, 5),
        }
        for feat, imp in list(feature_importances.items())[:6]
    ]

    recs = _TIER_RECOMMENDATIONS[tier]

    return CheckMeDecision(
        risk_tier=tier,
        risk_score=round(risk_prob, 4),
        confidence=confidence,
        clinical_flags=flags,
        top_features=top_features,
        recommendations=recs["recommendations"],
        next_steps=recs["next_steps"],
        disclaimer=_DISCLAIMER,
        raw_input=patient,
    )
