
### Breast Cancer Risk Stratification AI

> **⚕ CLINICAL DECISION SUPPORT ONLY** — Check Me does not diagnose breast cancer. All outputs require validation by a qualified clinician. Not for use as a sole basis for clinical action.

---

## What Is Check Me?

**Check Me** is a demo-ready, interpretable, and safety-first clinical AI tool for breast cancer risk stratification. It combines a logistic regression model trained on Wisconsin Breast Cancer Diagnostic-style features with a rule-based clinical safety flag system grounded in published oncology guidelines (NICE NG101, BI-RADS 5th Ed., ACS 2023).

It was built as a structured exploration of responsible healthcare AI — demonstrating how safety, interpretability, and clinical auditability can coexist with solid ML performance.

**Output:** Four risk tiers — `ROUTINE` / `ELEVATED` / `HIGH` / `URGENT` — each with actionable clinical recommendations and next steps.

---

## Live Demo

Open `checkme_desktop/index.html` in any browser. No server, no install required.

The desktop app includes five pages:

| Page | File | Description |
|------|------|-------------|
| Dashboard | `index.html` | Overview and quick-start |
| Assessment | `assess.html` | Step-by-step patient input form |
| Results | `results.html` | Risk score, flags, recommendations |
| Safety Flags | `flags.html` | All 8 clinical flags with reference table |
| Guidelines | `guidelines.html` | Accordion view of all referenced standards |

---

## Project Structure

```
checkme/
├── checkme_desktop/          # Multi-page desktop browser app (no server needed)
│   ├── index.html            # Dashboard
│   ├── assess.html           # Step-by-step assessment form
│   ├── results.html          # Risk results & recommendations
│   ├── flags.html            # Clinical safety flags reference
│   ├── guidelines.html       # Clinical guidelines accordion
│   ├── css/
│   │   └── app.css           # Shared styles (dark sidebar, red accent)
│   └── js/
│       └── model.js          # Shared model logic, state, sidebar builder
│
├── dataset.py                # Synthetic breast cancer dataset generator (n=3,000)
├── model.py                  # Model training, clinical flags, decision engine
├── api.py                    # FastAPI REST API
├── checkme_demo.html         # Standalone single-file browser demo
├── requirements.txt
└── model/
    ├── pipeline.pkl           # Trained scikit-learn pipeline
    └── metrics.json           # CV metrics + feature importance
```

---

## Architecture

The engine runs in three sequential layers:

```
Patient Input (FNA measurements + clinical risk factors)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 1 — CLINICAL SAFETY FLAGS                     │
│  Always evaluated first. Independent of ML.          │
│  8 flags grounded in NICE NG101, BI-RADS 5th Ed.    │
│  IMMEDIATE flag → forces URGENT tier                 │
│  PROMPT flag    → forces HIGH tier or above          │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 2 — LOGISTIC REGRESSION                       │
│  26 features: 15 FNA measurements + 11 clinical Rx   │
│  class_weight='balanced' — penalises missed cases    │
│  StandardScaler preprocessing                        │
│  Output: P(malignant) ∈ [0, 1]                       │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 3 — RISK TIER ASSIGNMENT                      │
│  URGENT:   P ≥ 0.72 OR any IMMEDIATE flag            │
│  HIGH:     P 0.48–0.72 OR any PROMPT flag            │
│  ELEVATED: P 0.28–0.48                               │
│  ROUTINE:  P < 0.28, no flags                        │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 4 — RECOMMENDATIONS + INTERPRETABILITY        │
│  Permutation feature importance (no black box)       │
│  Confidence assessment (distance from boundary)      │
│  Mandatory clinical disclaimer on every response     │
└──────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/checkme.git
cd checkme

# 2. Install
pip install -r requirements.txt

# 3. Generate dataset + train model
python dataset.py
python model.py

# 4. Run API
uvicorn api:app --reload --port 8000

# 5. Open demo (no server required)
open checkme_desktop/index.html
```

---

## API Reference

Base URL: `http://localhost:8000`

### `POST /assess`
Submit a patient profile for full risk assessment.

**Example request:**
```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58, "bmi": 30.1, "alcohol_drinks_week": 10,
    "radius_mean": 17.2, "texture_mean": 22.1, "perimeter_mean": 114.8,
    "area_mean": 932, "smoothness_mean": 0.108, "compactness_mean": 0.151,
    "concavity_mean": 0.168, "concave_points_mean": 0.091,
    "symmetry_mean": 0.194, "fractal_dim_mean": 0.064,
    "radius_worst": 21.4, "texture_worst": 30.1, "area_worst": 1438,
    "concavity_worst": 0.468, "concave_pts_worst": 0.187,
    "family_history_bc": 1, "prior_biopsy": 1, "hrt_use": 1,
    "brca_mutation": 0, "dense_breast": 1, "palpable_lump": 1,
    "nipple_discharge": 0, "skin_changes": 0
  }'
```

**Example response:**
```json
{
  "risk_tier": "HIGH",
  "probability": 0.63,
  "confidence": "moderate",
  "clinical_flags": ["fh_lump"],
  "recommendation": "Urgent breast clinic within 2 weeks. Consider MRI.",
  "next_steps": ["2-week-wait referral", "Triple assessment protocol"],
  "top_features": [["palpable_lump", 0.18], ["family_history_bc", 0.14]],
  "disclaimer": "This output is for clinical decision support only..."
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model-info` | Model metadata + CV performance metrics |
| `GET` | `/clinical-flags` | All 8 safety flags with rationale + guidelines |
| `GET` | `/example` | Three pre-built patients: routine / high / urgent |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## Model Performance

5-fold stratified cross-validation on synthetic dataset (n=3,000):

| Metric | Score | Notes |
|--------|-------|-------|
| ROC-AUC | 1.000 | Near-perfect on synthetic data |
| F1 | 0.999 | Balanced precision/recall |
| Recall | 0.998 | Sensitivity — minimising missed malignancies |
| Precision | 1.000 | Very few false positives |
| Brier Score | 0.0001 | Excellent calibration |

> **Note:** Near-perfect scores reflect the synthetic dataset's clean feature separation. Real-world performance on clinical data would be materially lower and requires rigorous prospective validation before any clinical use.

---

## Dataset

Synthetic dataset (n=3,000) based on:
- **UCI Wisconsin Breast Cancer Diagnostic** dataset feature distributions (Wolberg et al., 1995)
- **Clinical risk factor distributions** from ACS 2023, NICE NG101, and published epidemiology
- Malignant prevalence: ~35% (reflects a screening-positive referral population)

**26 features across two groups:**

*FNA Nucleus Measurements (15):*
`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`, `fractal_dimension` — mean and worst-case values.

*Clinical Risk Factors (11):*
`age`, `bmi`, `alcohol_drinks_week`, `family_history_bc`, `prior_biopsy`, `hrt_use`, `brca_mutation`, `dense_breast`, `palpable_lump`, `nipple_discharge`, `skin_changes`

---

## Clinical Safety Flags

Eight hard-coded rules that always override the ML layer:

| Flag | Condition | Urgency | Clinical Basis |
|------|-----------|---------|----------------|
| `brca_palpable` | BRCA mutation + palpable lump | IMMEDIATE | ACS: immediate surgical referral |
| `skin_nipple` | Skin changes + nipple discharge | IMMEDIATE | BI-RADS: inflammatory BC association |
| `skin_change` | Skin dimpling / peau d'orange | IMMEDIATE | Classic malignancy sign |
| `brca_known` | Known BRCA1/2 mutation | PROMPT | NICE NG101: high-risk surveillance |
| `fh_lump` | Family history + palpable lump | PROMPT | ACS: urgent imaging referral |
| `high_concavity` | Concavity worst > 0.70 | PROMPT | Cytological malignancy marker |
| `large_area` | Area worst > 2000 µm² | PROMPT | Strong malignancy predictor |
| `age_brca` | Age > 30 + BRCA mutation | PROMPT | NICE: annual MRI from age 30 |

---

## Risk Tiers & Clinical Actions

| Tier | ML Probability | Recommended Action |
|------|---------------|-------------------|
| **ROUTINE** | < 28% | Continue standard screening programme |
| **ELEVATED** | 28–48% | Supplemental imaging discussion; GP referral within 6 weeks |
| **HIGH** | 48–72% | Urgent breast clinic within 2 weeks; MRI consideration |
| **URGENT** | ≥ 72% or IMMEDIATE flag | 2-week-wait cancer pathway; triple assessment protocol |

---

## Responsible AI Practices

| Practice | Implementation |
|----------|----------------|
| Safety-first | Clinical flags always evaluated independently of ML |
| Recall-prioritised | `class_weight='balanced'` — penalises missed malignancies |
| Interpretable | Logistic regression; coefficients fully inspectable |
| Calibrated | Brier score monitored; well-calibrated probability outputs |
| Confidence reporting | Distance-from-boundary metric returned with every result |
| Mandatory disclaimer | Injected into every API response and UI screen |
| Auditable | All 8 safety flags cite published clinical guidelines |
| No black box | Full feature transparency + permutation importance scores |

---

## Clinical Guidelines Referenced

- **ACS 2023** — American Cancer Society Breast Cancer Screening Guidelines
- **NICE NG101** — Familial Breast Cancer (updated 2023)
- **BI-RADS 5th Edition** — ACR Breast Imaging Reporting and Data System
- **WHO Breast Cancer Fact Sheet 2023**
- **UCI Wisconsin BC Diagnostic Dataset** — Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995)

---

## Dependencies

```
fastapi
uvicorn
scikit-learn
numpy
pandas
pydantic
```

Frontend: pure HTML/CSS/JS — no framework, no build step, no dependencies.

---

## Limitations & Future Work

**Current limitations:**
- Trained entirely on synthetic data — clinical validation on real patient data is required before any deployment
- Logistic regression may underperform on complex non-linear patterns compared to ensemble methods
- FNA measurements require prior biopsy — not available for all screening patients
- No demographic fairness audit has been conducted

**Planned improvements:**
- [ ] Replace synthetic dataset with de-identified clinical data
- [ ] Add fairness metrics across age, BMI, and HRT-use subgroups
- [ ] XGBoost comparison with SHAP explainability
- [ ] FHIR-compatible API input schema
- [ ] Electron wrapper for true native desktop packaging

---

## Disclaimer

Check Me is a demonstration of responsible AI development practices in a healthcare context. It is **not a medical device**, **not CE/FDA cleared**, and **must not be used** as a substitute for clinical judgement, professional diagnosis, or established screening protocols.

All risk outputs are probabilistic estimates based on a synthetic dataset and should be interpreted by a qualified clinician in the context of the full patient presentation.

---

*Built with structured problem-solving, safety-first modeling, and clean engineering in mind.*
