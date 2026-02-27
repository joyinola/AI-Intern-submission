"""
Check Me — Clinical Safety Flags
==================================
Hard clinical rules grounded in published oncology guidelines.
These are ALWAYS evaluated independently of the ML model and surface
to the clinician regardless of model score. IMMEDIATE flags escalate
to RED tier; PROMPT flags escalate to YELLOW tier or above.

Clinical references:
  - ACS Breast Cancer Screening Guidelines (2023)
  - BI-RADS 5th Edition (ACR)
  - NICE NG101 Familial Breast Cancer
"""

from dataclasses import dataclass


@dataclass
class ClinicalFlag:
    id: str
    condition: str
    rationale: str
    urgency: str  # "IMMEDIATE" | "PROMPT" | "MONITOR"


CLINICAL_FLAGS: list[ClinicalFlag] = [
    ClinicalFlag(
        "brca_palpable",
        "Known BRCA mutation + palpable lump",
        "ACS: immediate surgical referral recommended",
        "IMMEDIATE",
    ),
    ClinicalFlag(
        "skin_nipple",
        "Skin changes + nipple discharge",
        "BI-RADS: combined signs associated with inflammatory BC",
        "IMMEDIATE",
    ),
    ClinicalFlag(
        "skin_change",
        "Skin dimpling or peau d'orange",
        "Classic sign of underlying malignancy",
        "IMMEDIATE",
    ),
    ClinicalFlag(
        "brca_known",
        "Known BRCA1/2 pathogenic variant",
        "NICE NG101: high-risk surveillance pathway",
        "PROMPT",
    ),
    ClinicalFlag(
        "fh_lump",
        "Strong family history + palpable lump",
        "ACS: combination warrants urgent imaging referral",
        "PROMPT",
    ),
    ClinicalFlag(
        "high_concavity",
        "Worst concavity > 0.7 (FNA)",
        "High nuclear concavity — cytological malignancy marker",
        "PROMPT",
    ),
    ClinicalFlag(
        "large_area",
        "Worst nuclear area > 2000 µm²",
        "Markedly enlarged nuclei — strong malignancy predictor",
        "PROMPT",
    ),
    ClinicalFlag(
        "age_brca",
        "Age > 30 with BRCA mutation",
        "NICE: annual MRI from age 30 for BRCA carriers",
        "PROMPT",
    ),
]

_FLAG_BY_ID: dict[str, ClinicalFlag] = {f.id: f for f in CLINICAL_FLAGS}


def evaluate_clinical_flags(row: dict) -> list[ClinicalFlag]:
    """
    Evaluate the patient dict against all clinical safety rules.
    Returns a deduplicated list of triggered ClinicalFlag objects.
    """
    triggered_ids: list[str] = []

    if row.get("brca_mutation") and row.get("palpable_lump"):
        triggered_ids.append("brca_palpable")
    if row.get("skin_changes") and row.get("nipple_discharge"):
        triggered_ids.append("skin_nipple")
    if row.get("skin_changes"):
        triggered_ids.append("skin_change")
    if row.get("brca_mutation"):
        triggered_ids.append("brca_known")
    if row.get("family_history_bc") and row.get("palpable_lump"):
        triggered_ids.append("fh_lump")
    if row.get("worst_concavity", 0) > 0.70:
        triggered_ids.append("high_concavity")
    if row.get("worst_area", 0) > 2000:
        triggered_ids.append("large_area")
    if row.get("brca_mutation") and row.get("age", 0) > 30:
        triggered_ids.append("age_brca")

    seen: set[str] = set()
    result: list[ClinicalFlag] = []
    for fid in triggered_ids:
        if fid not in seen:
            seen.add(fid)
            result.append(_FLAG_BY_ID[fid])
    return result
