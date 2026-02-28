# Check Me — Breast Cancer Risk Triage

> ⚕ **This is not a diagnostic tool.** Check Me is a self-screening support app. It cannot tell you whether you have cancer. For any health concerns, please consult a qualified healthcare professional.

---

## What It Does

Check Me takes a short health questionnaire and returns a colour-coded risk band — **GREEN**, **YELLOW**, or **RED** — with plain-language guidance on what to do next.

It combines two independent layers:
- A **Gradient Boosting model** trained on the real UCI Wisconsin Breast Cancer Diagnostic dataset (569 real FNA biopsy samples)
- A **clinical safety flag system** with 8 hard-coded rules grounded in ACS 2023, NICE NG101, and BI-RADS 5th Edition

The flags always run independently of the ML model. They can only escalate a result — they can never reduce it.

---

## Quick Start

### Option 1 — Docker (full app)

```bash
# Train the model locally first (only needed once)
python train_and_export.py

# Start the app
docker compose up --build
```

- App → http://localhost:8080
- API docs → http://localhost:8080/api/docs

The first build takes 1–2 minutes. After that, `docker compose up` starts in under 10 seconds — training never happens inside the container.

### Retrain the model

```bash
pip install -r requirements.txt
python train_and_export.py
```

This retrains on the UCI dataset and saves `model/artefacts/pipeline.joblib` and `model/artefacts/metrics.json`.

---

## Project Structure

```
├── model/
│   ├── core/
│   │   ├── trainer.py        # GradientBoosting training + permutation importance
│   │   ├── decision.py       # Band assignment (ML prob + CRS + flags → GREEN/YELLOW/RED)
│   │   ├── flags.py          # 8 clinical safety rules
│   │   ├── features.py       # 26 feature names and descriptions
│   │   └── persistence.py    # save/load pipeline.joblib + metrics.json
│   ├── dataset.py            # UCI loader + clinical feature augmentation
│   ├── train.py              # Training CLI entry point
│   └── artefacts/
│       ├── pipeline.joblib   # Trained model
│       └── metrics.json      # CV metrics + feature importance
│
├── api/
│   ├── main.py               # FastAPI app + startup model loading
│   ├── routers/assessment.py # POST /assess, GET /example
│   ├── schemas.py            # Pydantic input/output validation
│   └── dependencies.py       # Cached model loader
│
├── ui/
│   ├── index.html            # Dashboard
│   ├── assess.html           # Step-by-step assessment form (12 steps, 26 features)
│   ├── results.html          # Results — band, flags, recommendations, feature drivers
│   ├── flags.html            # Clinical safety flags reference
│   ├── guidelines.html       # Guidelines and model provenance
│   ├── css/app.css
│   └── js/model.js           # API client, JS fallback model, state management
│
├── notebooks/
│   └── checkme_eda_model.ipynb   # EDA, LR baseline, GBM, explainability (32 cells)
│
├── train_and_export.py       # Local training script
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── nginx/nginx.conf
├── requirements.txt
└── DATASET.md                # Full dataset and augmentation documentation
```

---

## How Scoring Works

Each assessment runs three checks in combination:

```
1. Safety Flags (8 rules)     — always evaluated first, independent of model
   IMMEDIATE flag → forces RED
   PROMPT flag    → minimum YELLOW

2. GBM Model                  — outputs P(malignant) from 0 to 1

3. Clinical Risk Score (CRS)  — integer 0–12 from observable symptoms
   Palpable lump +3 | Skin changes +3 | BRCA mutation +3
   Nipple discharge +2 | Family history +2 | Dense breast +1 ...

Band assignment:
  RED    → ML ≥ 0.85, or (ML ≥ 0.50 and CRS ≥ 3), or any IMMEDIATE flag
  YELLOW → ML ≥ 0.50, or CRS ≥ 3, or any PROMPT flag
  GREEN  → ML < 0.50 and CRS < 3 and no flags
```

---

## Model Performance

Trained on the **real UCI Wisconsin Breast Cancer Diagnostic dataset** — 569 patients, 37.3% malignant. 3-fold stratified cross-validation:

| Metric | Score |
|---|---|
| ROC-AUC | 0.993 ± 0.005 |
| Recall | 93.9% |
| Precision | 97.7% |
| F1 Score | 95.6% |
| Brier Score | 0.0002 |

**Model:** `GradientBoostingClassifier` (sklearn) — same gradient boosting algorithm as XGBoost. Swap by replacing the one import in `model/core/trainer.py`.

**Top predictor:** `worst_area` — the nucleus area of the largest cell in the FNA sample.

---

## Dataset

The training data is a hybrid of two layers:

| Layer | Source | Type |
|---|---|---|
| 15 FNA features | UCI Wisconsin Breast Cancer Diagnostic (Wolberg et al., 1995) | **Real measured data** |
| 11 clinical features | Synthetic, calibrated to ACS 2023, IARC 2012, NICE NG101 | Synthetic |

The 569 patients and their FNA measurements are real. The clinical columns (age, BMI, family history, BRCA status, etc.) are synthetically generated with class-conditional distributions matching published epidemiological statistics.

See [`DATASET.md`](DATASET.md) for the full breakdown of every feature and its source.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/assess` | POST | Full risk assessment — returns band, score, flags, recommendations |
| `/api/example` | GET | Three pre-built patients (green / yellow / red) |
| `/api/health` | GET | Docker health check |
| `/api/docs` | GET | Swagger UI |

**Example request:**

```bash
curl -X POST http://localhost:8080/api/assess \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "bmi": 28.5, "alcohol_drinks_week": 7,
    "mean_radius": 15.0, "mean_texture": 20.5, "mean_perimeter": 98.0,
    "mean_area": 720, "mean_smoothness": 0.10, "mean_compactness": 0.11,
    "mean_concavity": 0.09, "mean_concave_points": 0.05,
    "mean_symmetry": 0.18, "mean_fractal_dimension": 0.063,
    "worst_radius": 18.2, "worst_texture": 27.0, "worst_area": 1050,
    "worst_concavity": 0.32, "worst_concave_points": 0.14,
    "family_history_bc": 1, "prior_biopsy": 1, "hrt_use": 1,
    "brca_mutation": 0, "dense_breast": 1, "palpable_lump": 0,
    "nipple_discharge": 0, "skin_changes": 0
  }'
```

---

## Clinical Safety Flags

| Flag | Trigger | Urgency | Source |
|---|---|---|---|
| `brca_palpable` | BRCA mutation + palpable lump | IMMEDIATE → RED | ACS 2023 |
| `skin_nipple` | Skin changes + nipple discharge | IMMEDIATE → RED | BI-RADS 5th Ed. |
| `skin_change` | Skin dimpling / peau d'orange | IMMEDIATE → RED | BI-RADS 5th Ed. |
| `brca_known` | Known BRCA1/2 mutation | PROMPT → min. YELLOW | NICE NG101 |
| `fh_lump` | Family history + palpable lump | PROMPT → min. YELLOW | ACS 2023 |
| `high_concavity` | Worst concavity > 0.70 | PROMPT → min. YELLOW | BI-RADS 5th Ed. |
| `large_area` | Worst nucleus area > 2,000 µm² | PROMPT → min. YELLOW | BI-RADS 5th Ed. |
| `age_brca` | BRCA mutation + age > 30 | PROMPT → min. YELLOW | NICE NG101 |

---

## Limitations

- Clinical features are synthetically generated — not from the same 569 UCI patients
- n=569 is small for a production clinical model
- No external or prospective validation has been performed
- FNA measurements (the strongest predictors) require a prior biopsy — most self-screening users will not have them
- The JS offline fallback is a linear approximation and cannot fully replicate GBM non-linear behaviour

---

## References

- Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI ML Repository. https://doi.org/10.24432/C5DW2B
- American Cancer Society. *Breast Cancer Screening Guidelines* (2023)
- NICE. *Familial breast cancer: NG101* (updated 2023). https://www.nice.org.uk/guidance/ng101
- American College of Radiology. *BI-RADS 5th Edition* (2013)
