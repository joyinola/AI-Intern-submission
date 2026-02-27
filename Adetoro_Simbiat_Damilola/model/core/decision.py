"""
Check Me — Decision & Recommendation Logic
============================================
Combines GradientBoosting probability, clinical safety flags, clinical
risk score, and band thresholds into a structured CheckMeDecision.

Risk Band System (consumer self-screening):
  ┌────────┬──────────────────────────────┬───────────────────────────────────┐
  │ Band   │ Trigger                      │ Meaning                           │
  ├────────┼──────────────────────────────┼───────────────────────────────────┤
  │ GREEN  │ ML < 0.50 AND crs < 3 AND    │ Low concern — routine care        │
  │        │ no flags                     │                                   │
  │ YELLOW │ ML < 0.50 AND crs >= 3, OR   │ Some risk factors — worth a check │
  │        │ ML >= 0.50, OR PROMPT flag   │                                   │
  │ RED    │ ML >= 0.50 AND crs >= 3, OR  │ Multiple signals — see a doctor   │
  │        │ ML >= 0.85, OR IMMEDIATE     │                                   │
  └────────┴──────────────────────────────┴───────────────────────────────────┘

Clinical Risk Score (crs) — integer 0-12:
  palpable_lump(3) + skin_changes(3) + nipple_discharge(2) + brca_mutation(3)
  + family_history(2) + dense_breast(1) + hrt_use(1) + prior_biopsy(1)
  + age>=50(1) + age>=60(1) + alcohol>=7(1)

Design rationale:
  - The GB model is trained on FNA cytology — it is near-binary (P~0 or P~0.99).
  - For a consumer self-screening app, clinical risk factors matter even when
    FNA looks benign. The clinical risk score bridges this gap.
  - Flags can only escalate, never downgrade.
  - Language is calm, supportive, and never diagnostic.

References: ACS 2023, NICE NG101, BI-RADS 5th Edition
"""

from dataclasses import dataclass, field
import pandas as pd
from .features import FEATURE_COLS, FEATURE_DESCRIPTIONS
from .flags import ClinicalFlag, evaluate_clinical_flags


# ── Recommendation content ─────────────────────────────────────────────────────

_TIER_RECOMMENDATIONS: dict[str, dict] = {
    "GREEN": {
        "color":   "#27ae60",
        "label":   "Green",
        "sub":     "Low Concern — Keep Up the Good Work",
        "recommendations": [
            "Keep up with your regular health check-ups and annual mammography if you are 40 or older",
            "Do a monthly self-check — it helps you get familiar with what is normal for you",
            "Maintaining a healthy weight and limiting alcohol can help reduce long-term breast health risk",
            "Share any changes in your family health history with your doctor at your next visit",
        ],
        "next_steps": [
            "Book your routine mammogram if you haven't had one in the past year (age 40+)",
            "Ask your GP about a clinical breast exam at your next check-up",
            "Come back and re-assess if you notice any new changes",
        ],
    },
    "YELLOW": {
        "color":   "#f5a623",
        "label":   "Yellow",
        "sub":     "Some Risk Factors Present — Worth a Check-Up",
        "recommendations": [
            "You have some risk factors that are worth discussing with your doctor — this does not mean anything is wrong",
            "A check-up or clinical breast exam can give you peace of mind and catch anything early",
            "Keep track of any changes — like lumps, skin texture, or nipple changes — and note when they started",
            "Reducing alcohol and maintaining a healthy weight are positive steps you can take right now",
        ],
        "next_steps": [
            "Book an appointment with your GP to discuss your risk factors — aim for within the next few weeks",
            "Ask about whether supplemental screening (like ultrasound) makes sense for you",
            "If you have a family history of breast cancer, ask your GP about genetic counselling",
        ],
    },
    "RED": {
        "color":   "#c0392b",
        "label":   "Red",
        "sub":     "Please See a Doctor Soon",
        "recommendations": [
            "This result suggests it would be a good idea to see a doctor soon — early attention gives the best outcomes",
            "Make an appointment with your GP or a breast clinic as soon as possible — don't put it off",
            "Write down your symptoms and when you first noticed them so you can share them clearly",
            "Bring a trusted friend or family member to your appointment for support",
        ],
        "next_steps": [
            "Contact your GP or a breast clinic to arrange an appointment — aim for within the next 1–2 weeks",
            "Ask your doctor about a clinical breast exam and whether imaging is recommended",
            "If you are worried while waiting, most hospitals have a nurse helpline you can call",
            "Remember: seeing a doctor early is the right and brave thing to do",
        ],
    },
}

_DISCLAIMER = (
    "This is not a diagnosis. Check Me is a self-screening support tool only. "
    "It cannot tell you whether you have or don't have cancer. "
    "For any concerns, please consult a healthcare professional."
)


# ── Output dataclass ───────────────────────────────────────────────────────────

@dataclass
class CheckMeDecision:
    risk_tier: str
    risk_score: float
    clinical_risk_score: int
    confidence: str
    clinical_flags: list[ClinicalFlag]
    top_features: list[dict]
    recommendations: list[str]
    next_steps: list[str]
    tier_color: str
    tier_label: str
    tier_sub: str
    disclaimer: str
    raw_input: dict = field(default_factory=dict)


# ── Clinical risk score ────────────────────────────────────────────────────────

def _clinical_risk_score(patient: dict) -> int:
    """
    Rule-based clinical risk score (0–12).
    Independent of the ML model — based on published risk factor weights.
    """
    score = 0
    score += 3 if patient.get("palpable_lump")     else 0
    score += 3 if patient.get("skin_changes")       else 0
    score += 2 if patient.get("nipple_discharge")   else 0
    score += 3 if patient.get("brca_mutation")      else 0
    score += 2 if patient.get("family_history_bc")  else 0
    score += 1 if patient.get("dense_breast")       else 0
    score += 1 if patient.get("hrt_use")            else 0
    score += 1 if patient.get("prior_biopsy")       else 0
    age = patient.get("age", 0)
    score += 1 if age >= 50 else 0
    score += 1 if age >= 60 else 0
    score += 1 if patient.get("alcohol_drinks_week", 0) >= 7 else 0
    return score


# ── Core decision function ────────────────────────────────────────────────────

def make_decision(patient: dict, pipeline, feature_importances: dict) -> CheckMeDecision:
    """
    Full decision pipeline for one patient.

    Steps:
      1. Get ML probability from GradientBoosting.
      2. Compute clinical risk score from rule-based factors.
      3. Evaluate clinical safety flags.
      4. Determine band (flags and crs can only escalate, never downgrade).
      5. Compute confidence.
      6. Return structured CheckMeDecision.
    """
    # 1. ML probability
    X_in = pd.DataFrame([{col: patient.get(col, 0) for col in FEATURE_COLS}])
    risk_prob = float(pipeline.predict_proba(X_in)[0, 1])

    # 2. Clinical risk score
    crs = _clinical_risk_score(patient)

    # 3. Clinical flags
    flags = evaluate_clinical_flags(patient)
    has_immediate = any(f.urgency == "IMMEDIATE" for f in flags)
    has_prompt    = any(f.urgency == "PROMPT"    for f in flags)

    # 4. Band assignment — more nuanced for consumer use
    if has_immediate or risk_prob >= 0.85:
        tier = "RED"
    elif (risk_prob >= 0.50 and crs >= 3) or has_prompt:
        tier = "RED"
    elif risk_prob >= 0.50 or crs >= 3:
        tier = "YELLOW"
    else:
        tier = "GREEN"

    # 5. Confidence
    if has_immediate or risk_prob >= 0.85:
        confidence = "HIGH"
    elif risk_prob >= 0.50 or crs >= 6:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    # 6. Top features
    top_features = [
        {
            "feature":     feat,
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            "value":       round(patient.get(feat, 0), 4),
            "importance":  round(imp, 5),
        }
        for feat, imp in list(feature_importances.items())[:6]
    ]

    cfg = _TIER_RECOMMENDATIONS[tier]

    return CheckMeDecision(
        risk_tier=tier,
        risk_score=round(risk_prob, 4),
        clinical_risk_score=crs,
        confidence=confidence,
        clinical_flags=flags,
        top_features=top_features,
        recommendations=cfg["recommendations"],
        next_steps=cfg["next_steps"],
        tier_color=cfg["color"],
        tier_label=cfg["label"],
        tier_sub=cfg["sub"],
        disclaimer=_DISCLAIMER,
        raw_input=patient,
    )
