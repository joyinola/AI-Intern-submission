# Dataset Source & Construction

Due to lack of public datasets mapping directly to symptom-based breast cancer triage, a synthetic dataset was constructed.

## Feature Logic
Risk increases with:
- Age
- Family history
- Previous lumps
- Lump size
- Skin dimpling
- Nipple discharge
- Weight loss
- Fatigue

Risk decreases with:
- Pregnancy

## Noise Injection
Gaussian noise added to simulate real-world uncertainty.

## Sample Size
12,000 synthetic patients
