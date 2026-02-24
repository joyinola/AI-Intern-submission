# Check Me — AI Risk Triage Demo

## Setup
```bash
pip install -r requirements.txt
```

## Generate Dataset
```bash
python generate_data.py
```

## Train Model
Run `notebook.ipynb`

## Run API
```bash
uvicorn app:app --reload
```

## Endpoint

`POST /predict`

### Example: Test Green Case (Low Risk)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 25,
    "age_group": "young",
    "family_history": 0,
    "previous_lumps": 0,
    "breast_pain": 0,
    "nipple_discharge": 0,
    "skin_dimples": 0,
    "lump_size_mm": 0.0,
    "symptom_duration_days": 0,
    "pregnancy_status": 0,
    "hormonal_contraception": 1,
    "fever": 0,
    "weight_loss": 0,
    "fatigue": 0,
    "region": "urban",
    "language": "en"
  }'
```

### Example: Test Red Case (High Risk)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 65,
    "age_group": "senior",
    "family_history": 1,
    "previous_lumps": 1,
    "breast_pain": 0,
    "nipple_discharge": 1,
    "skin_dimples": 1,
    "lump_size_mm": 25.5,
    "symptom_duration_days": 45,
    "pregnancy_status": 0,
    "hormonal_contraception": 0,
    "fever": 0,
    "weight_loss": 1,
    "fatigue": 1,
    "region": "rural",
    "language": "rw"
  }'
```

### Example: Test Yellow Case (Medium Risk)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 42,
    "age_group": "adult",
    "family_history": 0,
    "previous_lumps": 1,
    "breast_pain": 1,
    "nipple_discharge": 0,
    "skin_dimples": 0,
    "lump_size_mm": 5.0,
    "symptom_duration_days": 14,
    "pregnancy_status": 0,
    "hormonal_contraception": 1,
    "fever": 0,
    "weight_loss": 0,
    "fatigue": 1,
    "region": "urban",
    "language": "fr"
  }'
```