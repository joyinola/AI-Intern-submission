import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
N = 12000

# 1. Feature Generation (Symptom prevalence)
age = np.random.randint(18, 80, N)
family_history = np.random.binomial(1, 0.25, N)
previous_lumps = np.random.binomial(1, 0.15, N)
breast_pain = np.random.binomial(1, 0.4, N)
nipple_discharge = np.random.binomial(1, 0.1, N)
skin_dimples = np.random.binomial(1, 0.08, N)

# Lump logic: size is 0 if no lump is present
lump_present = np.random.binomial(1, 0.35, N)
lump_size_mm = np.where(
    lump_present == 1,
    np.random.normal(18, 10, N).clip(2, 60),
    0
)

symptom_duration_days = np.random.randint(1, 120, N)
pregnancy_status = np.random.binomial(1, 0.08, N)
hormonal_contraception = np.random.binomial(1, 0.3, N)

# Systemic symptoms (Optional extra features)
fever = np.random.binomial(1, 0.12, N)
weight_loss = np.random.binomial(1, 0.1, N)
fatigue = np.random.binomial(1, 0.25, N)

# Categorical features
region = np.random.choice(["urban", "rural"], N, p=[0.6, 0.4])
language = np.random.choice(["en", "fr", "rw"], N, p=[0.5, 0.2, 0.3])

# Binned features for model variety
age_group = pd.cut(age, bins=[17, 30, 45, 60, 100],
                    labels=["young", "adult", "middle", "senior"])

# We add a strong negative intercept (-8.0) to make risk rare.
# High weights are assigned to critical clinical indicators.
logit = (
    -8.0 +                          # Strong healthy baseline (intercept)
    0.04 * age +                    # Age increases risk gradually
    2.2 * family_history +          # Strong genetic driver
    1.5 * previous_lumps +
    0.06 * lump_size_mm +           # Larger masses carry higher weight
    2.5 * skin_dimples +            # Highly specific indicator
    2.0 * nipple_discharge +        # Critical symptom
    0.01 * symptom_duration_days +
    0.8 * weight_loss +
    np.random.normal(0, 1.2, N)     # Gaussian noise to prevent perfect separability
)

# Sigmoid transformation to obtain probabilities
prob = 1 / (1 + np.exp(-logit))

# Label as 1 (High Risk) if probability > 0.5
label = (prob > 0.5).astype(int)

# 3. Create DataFrame
df = pd.DataFrame({
    "age": age,
    "age_group": age_group.astype(str),
    "family_history": family_history,
    "previous_lumps": previous_lumps,
    "breast_pain": breast_pain,
    "nipple_discharge": nipple_discharge,
    "skin_dimples": skin_dimples,
    "lump_size_mm": lump_size_mm,
    "symptom_duration_days": symptom_duration_days,
    "pregnancy_status": pregnancy_status,
    "hormonal_contraception": hormonal_contraception,
    "fever": fever,
    "weight_loss": weight_loss,
    "fatigue": fatigue,
    "region": region,
    "language": language,
    "target": label
})

# 4. Save and Verify
df.to_csv("data/synthetic_data.csv", index=False)
print(f"Dataset saved: {df.shape}")
print("New Class Balance:")
print(df["target"].value_counts(normalize=True))