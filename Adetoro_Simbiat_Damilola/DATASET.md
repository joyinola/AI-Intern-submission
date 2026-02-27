# Check Me — Dataset Documentation

## Overview

The Check Me model is trained on a **hybrid dataset** combining two distinct data sources:

| Layer | Source | Type | Rows | Columns |
|---|---|---|---|---|
| FNA Features | UCI Wisconsin Breast Cancer Diagnostic (1995) | **Real measured data** | 569 | 15 |
| Clinical Features | Synthetic generation from published epidemiology | **Synthetic** | 569 | 11 |
| **Final Dataset** | Combined | Hybrid | **569** | **26 + 1 label** |

The 569 patients and their FNA measurements are real. The 11 clinical columns attached to those patients are not — they are generated using probability distributions calibrated to published epidemiological studies.

---

## Part 1 — The Real Data: UCI Wisconsin FNA Features

### What it is

The **UCI Wisconsin Breast Cancer Diagnostic Dataset** was collected at the University of Wisconsin Hospital by Dr. William Wolberg. Each row represents one patient who had a **fine needle aspiration (FNA)** biopsy of a breast mass. A pathologist digitised images of the aspirated cells and computed 30 numerical measurements describing the shape and texture of the cell nuclei.

- **569 patients** — 212 malignant (37.3%), 357 benign (62.7%)
- **Label**: confirmed biopsy outcome (malignant or benign)
- **No missing values, no duplicates**
- Available via `sklearn.datasets.load_breast_cancer()`

> **Citation**: Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

### Important: sklearn target encoding

sklearn encodes the target as `0 = malignant, 1 = benign` (counterintuitive). The dataset module **inverts** this so that `malignant = 1` throughout the codebase:

```python
df["malignant"] = (bc.target == 0).astype(int)
```

### The 30 original UCI features (grouped)

The UCI dataset provides 10 measurements per cell nucleus, each computed three ways — giving 30 features total. We use **15** of these (mean + worst; SE features are dropped as noisier):

| Group | Features Used | Features Dropped | Reason |
|---|---|---|---|
| **Mean** (avg across all cells) | 10 features ✓ | — | Most representative central tendency |
| **Worst** (largest value seen) | 5 features ✓ | 5 removed | Most clinically alarming cell retained |
| **SE** (standard error within sample) | None ✗ | 10 removed | Noisier; correlated with mean; adds variance |

### The 15 retained FNA features

```
mean_radius              Mean cell nucleus radius
mean_texture             Mean nucleus texture (grey-scale SD)
mean_perimeter           Mean nucleus perimeter
mean_area                Mean nucleus area (µm²)
mean_smoothness          Mean contour smoothness
mean_compactness         Mean compactness (perimeter² / area)
mean_concavity           Mean severity of concave portions
mean_concave_points      Mean number of concave points
mean_symmetry            Mean nucleus symmetry
mean_fractal_dimension   Mean fractal dimension (coastline approximation)

worst_radius             Largest nucleus radius seen in sample
worst_texture            Worst-case nucleus texture
worst_area               Largest nucleus area (µm²)
worst_concavity          Worst-case nucleus concavity
worst_concave_points     Worst-case concave nucleus points
```

### Why concavity features matter most

The top predictors are all concavity-related. Malignant cells have **irregular, deeply indented nuclear membranes** — the membrane buckles inward as the cell divides abnormally. Benign cells have smooth, round nuclei. This is a direct physical consequence of cancer biology, which is why the model achieves near-perfect AUC on FNA data alone.

| Feature | Benign Mean | Malignant Mean | Ratio |
|---|---|---|---|
| `worst_concave_points` | 0.074 | 0.183 | 2.47× |
| `worst_concavity` | 0.175 | 0.448 | 2.56× |
| `worst_area` | 558.9 µm² | 1422.3 µm² | 2.55× |
| `mean_concavity` | 0.046 | 0.161 | 3.50× |

---

## Part 2 — The Synthetic Data: Clinical Feature Augmentation

### Why augmentation is needed

The UCI dataset only contains FNA measurements — data that is only available *after* a biopsy has been taken. But Check Me is a **self-screening app for consumers** who have not yet had a biopsy. They know their age, whether they have a family history of breast cancer, whether they have noticed a lump. These factors don't exist in the UCI data.

To bridge this gap, 11 clinical and demographic features are generated synthetically for each of the 569 patients, using **class-conditional probability distributions grounded in published epidemiology**.

> ⚠️ **The clinical features are synthetic.** They are calibrated to real population statistics, but they have not been collected from real patients. A production clinical system would require a prospectively collected dataset with both FNA measurements and contemporaneous clinical risk factors from the same patients.

### How augmentation works

The core principle is **class-conditional generation**: malignant and benign patients get different probability distributions, matching the real-world difference in prevalence for each risk factor.

For binary features (yes/no):

```python
def coin(p_benign: float, p_malignant: float) -> np.ndarray:
    p = np.where(mal == 1, p_malignant, p_benign)
    return (rng.random(n) < p).astype(int)
```

For continuous features, a clipped normal distribution is used separately for each class.

A fixed random seed (`seed=2024`) ensures the augmentation is **fully reproducible** — running the code twice produces identical results.

---

## The 11 Synthetic Clinical Features

### 1. Age

```python
Malignant: Normal(mean=57, std=13), clipped [25, 90]
Benign:    Normal(mean=47, std=14), clipped [25, 90]
```

**Rationale**: Breast cancer risk increases substantially with age. The ACS reports a median diagnosis age of approximately 62. Malignant patients in the augmented dataset are therefore drawn from an older distribution than benign patients. The 10-year mean difference (57 vs 47) reflects real screening population demographics.

---

### 2. BMI (Body Mass Index)

```python
Malignant: Normal(mean=28.2, std=5.8), clipped [16, 55]
Benign:    Normal(mean=26.5, std=5.5), clipped [16, 55]
```

**Rationale**: Post-menopausal obesity increases circulating oestrogen levels, raising breast cancer risk. Adipose tissue converts androgens to oestrogen via aromatase. The malignant distribution is shifted slightly higher (mean 28.2 vs 26.5 kg/m²), reflecting this modest but established association.

---

### 3. Alcohol Drinks per Week

```python
Both classes: Exponential(scale=3.0), clipped [0, 30]
```

**Rationale**: The IARC Handbook (2012) found that each additional 10g of alcohol per day raises relative breast cancer risk by approximately 7–10%. The exponential distribution is appropriate here because alcohol consumption is heavily right-skewed in populations — most people drink little, a few drink a lot. This feature is intentionally not class-conditional because the effect size is modest and population-level distributions are similar; what matters is the threshold (≥7 drinks/week scores +1 point in the Clinical Risk Score).

---

### 4. Family History of Breast Cancer

```python
Malignant: p = 0.28  (28% have a first-degree relative with BC)
Benign:    p = 0.12  (12% have a first-degree relative with BC)
```

**Rationale**: Having a first-degree relative (mother, sister, or daughter) with breast cancer approximately doubles lifetime risk (ACS 2023). Population prevalence of a positive family history is approximately 12–15% in general screening populations. Malignant patients are assigned a 2.3× higher prevalence (28% vs 12%) to reflect this relative risk.

---

### 5. Prior Breast Biopsy

```python
Malignant: p = 0.30
Benign:    p = 0.15
```

**Rationale**: A prior biopsy showing atypical hyperplasia raises future breast cancer risk approximately 4-fold. Even without atypia, the fact of a prior biopsy indicates a prior clinical concern and is a meaningful risk marker. Malignant patients are assigned 2× the prior biopsy prevalence of benign patients.

---

### 6. Hormone Replacement Therapy (HRT) Use

```python
Malignant (age > 45): p = 0.32
Benign    (age > 45): p = 0.20
Both      (age ≤ 45): p = 0     (HRT is post-menopausal)
```

**Rationale**: Combined oestrogen-progestogen HRT increases breast cancer risk by approximately 20–25% (Women's Health Initiative trial, 2002). HRT is only relevant post-menopause (age > 45 used as a proxy for the augmentation). The conditional check `age > 45` ensures HRT is never assigned to young patients.

---

### 7. Known BRCA1/2 Mutation

```python
Malignant: p = 0.12  (12% carry a BRCA mutation)
Benign:    p = 0.04  (4% carry a BRCA mutation)
```

**Rationale**: BRCA1 mutation carriers have a 55–72% lifetime breast cancer risk; BRCA2 carriers have 45–69% (National Cancer Institute). General population BRCA carrier prevalence is approximately 0.2–0.5%, but in a screened high-risk population (which is who presents for biopsy) the prevalence is higher — around 3–5% in general screened populations, and significantly higher among those who have already developed breast cancer. The 3× relative prevalence in malignant cases (12% vs 4%) reflects this.

---

### 8. Dense Breast Tissue (BI-RADS C/D)

```python
Malignant: p = 0.48
Benign:    p = 0.30
```

**Rationale**: Dense breast tissue (classified as BI-RADS category C or D on mammography) is an independent risk factor for breast cancer, raising risk by approximately 1.5–2×. It also makes mammograms harder to read, as dense tissue and tumours both appear white. Approximately 40% of women who undergo mammography have dense tissue. Malignant patients are assigned higher prevalence (48% vs 30%) to reflect both the direct risk and the selection bias of the biopsy population.

---

### 9. Palpable Lump

```python
Malignant: p = 0.68
Benign:    p = 0.20
```

**Rationale**: A palpable breast lump is the most common presenting symptom of breast cancer — approximately 60–75% of breast cancer diagnoses are initially detected by the patient or clinician as a palpable mass. Benign masses (fibroadenomas, cysts) are also palpable, which is why the benign prevalence is still 20%. The 3.4× difference (68% vs 20%) is the largest class-conditional ratio of any binary feature, reflecting the strong diagnostic association.

---

### 10. Nipple Discharge

```python
Malignant: p = 0.25
Benign:    p = 0.08
```

**Rationale**: Spontaneous, non-lactational nipple discharge — particularly if unilateral, bloody, or from a single duct — is a clinical red flag. It can indicate intraductal papilloma or ductal carcinoma in situ (DCIS). Approximately 7–15% of breast cancer diagnoses involve nipple discharge as a presenting symptom. Benign discharge (bilateral, milky) is more common and less concerning; the 3.1× ratio (25% vs 8%) reflects the increased diagnostic significance in malignant cases.

---

### 11. Skin Changes (Dimpling / Peau d'Orange)

```python
Malignant: p = 0.20
Benign:    p = 0.04
```

**Rationale**: Skin dimpling and peau d'orange (orange-peel texture) are caused by cancer cells blocking dermal lymphatics, causing characteristic skin changes. These are classic clinical signs of locally advanced or inflammatory breast cancer (IBC). They are uncommon overall but strongly associated with malignancy when present — hence the 5× prevalence ratio (20% vs 4%). Any skin change triggers an IMMEDIATE safety flag in the Check Me system.

---

## Augmented Dataset Summary

| Feature | Type | Benign | Malignant | Source |
|---|---|---|---|---|
| `mean_radius` | continuous | real UCI | real UCI | Wolberg et al. 1995 |
| `worst_area` | continuous | real UCI | real UCI | Wolberg et al. 1995 |
| *(13 other FNA features)* | continuous | real UCI | real UCI | Wolberg et al. 1995 |
| `age` | continuous | N(47, 14) | N(57, 13) | ACS 2023 |
| `bmi` | continuous | N(26.5, 5.5) | N(28.2, 5.8) | WCRF/AICR 2018 |
| `alcohol_drinks_week` | continuous | Exp(3.0) | Exp(3.0) | IARC 2012 |
| `family_history_bc` | binary | p=0.12 | p=0.28 | ACS 2023 |
| `prior_biopsy` | binary | p=0.15 | p=0.30 | Dupont & Page 1985 |
| `hrt_use` | binary | p=0.20 | p=0.32 | WHI Trial 2002 |
| `brca_mutation` | binary | p=0.04 | p=0.12 | NCI BRCA data |
| `dense_breast` | binary | p=0.30 | p=0.48 | BI-RADS 5th Ed. |
| `palpable_lump` | binary | p=0.20 | p=0.68 | ACS clinical guidance |
| `nipple_discharge` | binary | p=0.08 | p=0.25 | ACS clinical guidance |
| `skin_changes` | binary | p=0.04 | p=0.20 | BI-RADS / ACS |

---

## How the Dataset is Built (Code Flow)

```
1. load_uci_base()
   └─ sklearn.datasets.load_breast_cancer()
   └─ Rename columns (spaces → underscores)
   └─ Invert target (0=malignant → 1=malignant)
   └─ Returns: DataFrame (569 rows × 31 cols)

2. augment_with_clinical_features(df)
   └─ For each clinical feature:
       └─ Generate values using class-conditional distribution
       └─ Clip to valid range
   └─ Append 11 new columns to DataFrame
   └─ Returns: DataFrame (569 rows × 42 cols)

3. train_model(df)
   └─ Selects the 26 FEATURE_COLS (15 FNA + 11 clinical)
   └─ Target: 'malignant' column
   └─ Trains GradientBoostingClassifier
```

The combined CSV is saved to `model/data/uci_breast_cancer.csv` for reproducibility. If the file already exists, it is loaded directly without re-generating.

---

## Limitations

- **Clinical features are not validated against real patient records.** The distributions are calibrated to published population statistics, but they were not collected from the same 569 UCI patients.
- **FNA features dominate the model** (top 5 predictors are all FNA measurements). The clinical features primarily serve the YELLOW band system via the independent Clinical Risk Score, not the ML model directly.
- **n=569 is small** for a clinical deployment model. The augmentation does not increase the number of independent observations — it only adds columns to existing rows.
- **No temporal validation** has been performed. The model was not tested on data from a different time period or clinical site.

---

## References

| Reference | Used For |
|---|---|
| Wolberg, Street & Mangasarian (1995). *UCI Breast Cancer Wisconsin (Diagnostic)*. | FNA dataset |
| American Cancer Society. *Cancer Facts & Figures 2023*. | Age, family history, palpable lump prevalence |
| IARC Handbook of Cancer Prevention, Vol. 15: *Breast Cancer Screening* (2012). | Alcohol relative risk |
| Rossouw et al. (2002). *Writing Group for the WHI. JAMA.* | HRT risk estimates |
| Dupont & Page (1985). *N Engl J Med.* | Prior biopsy / atypical hyperplasia risk |
| American College of Radiology. *BI-RADS 5th Edition* (2013). | Dense breast, skin changes |
| NICE NG101. *Familial Breast Cancer* (updated 2023). | BRCA mutation prevalence and surveillance |
| National Cancer Institute. *BRCA Gene Mutations: Cancer Risk and Genetic Testing* (2023). | BRCA lifetime risk estimates |
