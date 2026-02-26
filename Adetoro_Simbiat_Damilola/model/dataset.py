"""
Check Me — Breast Cancer Risk Dataset Generator
================================================
Synthetic dataset based on the UCI Wisconsin Breast Cancer Diagnostic dataset
feature distributions + additional clinical/demographic risk factors.

Real dataset reference: Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995)
UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Features include:
  - Cell nuclei measurements (from fine needle aspirate, FNA)
  - Clinical risk factors (age, family history, hormonal, lifestyle)
  - Imaging descriptors (texture, area, concavity, etc.)
"""

import numpy as np
import pandas as pd

np.random.seed(2024)


def generate_breast_cancer_dataset(n: int = 2500) -> pd.DataFrame:
    """
    Generate a synthetic breast cancer risk dataset.

    Label: malignant (1) vs benign (0)
    Prevalence set to ~35% malignant to reflect screening-positive population.
    """
    malignant = np.random.choice([0, 1], size=n, p=[0.65, 0.35])

    def feature(benign_mean, benign_std, mal_mean, mal_std, clip_lo=None, clip_hi=None):
        vals = np.where(
            malignant == 0,
            np.random.normal(benign_mean, benign_std, n),
            np.random.normal(mal_mean, mal_std, n),
        )
        if clip_lo is not None or clip_hi is not None:
            vals = np.clip(vals, clip_lo, clip_hi)
        return vals

    # ── Nucleus measurements (from Wisconsin dataset distributions) ──────────
    radius_mean         = feature(12.1, 1.8, 17.5, 3.5, 6, 30)
    texture_mean        = feature(17.9, 4.0, 21.6, 4.2, 9, 40)
    perimeter_mean      = feature(78.1, 11.5, 115.4, 22.0, 40, 200)
    area_mean           = feature(463, 134, 978, 368, 100, 2500)
    smoothness_mean     = feature(0.092, 0.013, 0.103, 0.016, 0.04, 0.17)
    compactness_mean    = feature(0.080, 0.034, 0.145, 0.058, 0.01, 0.35)
    concavity_mean      = feature(0.046, 0.038, 0.161, 0.079, 0.0, 0.45)
    concave_points_mean = feature(0.025, 0.018, 0.088, 0.031, 0.0, 0.20)
    symmetry_mean       = feature(0.174, 0.024, 0.193, 0.031, 0.1, 0.3)
    fractal_dim_mean    = feature(0.062, 0.007, 0.063, 0.008, 0.04, 0.10)

    # Worst (largest) nucleus measurements
    radius_worst        = feature(14.3, 2.6, 21.1, 4.5, 7, 36)
    texture_worst       = feature(25.7, 6.2, 29.3, 6.0, 12, 50)
    area_worst          = feature(559, 195, 1422, 567, 150, 4000)
    concavity_worst     = feature(0.135, 0.098, 0.448, 0.158, 0.0, 1.0)
    concave_pts_worst   = feature(0.074, 0.034, 0.182, 0.047, 0.0, 0.30)

    # ── Clinical risk factors ────────────────────────────────────────────────
    age = np.where(
        malignant == 0,
        np.random.normal(47, 14, n),
        np.random.normal(57, 13, n)
    ).clip(25, 90)

    # BRCA1/2 family history (increases malignancy risk ~2×)
    family_history_bc = np.where(
        malignant == 0,
        np.random.rand(n) < 0.12,
        np.random.rand(n) < 0.28,
    ).astype(int)

    # Prior biopsy
    prior_biopsy = np.where(
        malignant == 0,
        np.random.rand(n) < 0.15,
        np.random.rand(n) < 0.30,
    ).astype(int)

    # Hormonal factors
    hrt_use = np.where(
        malignant == 0,
        (age > 45) & (np.random.rand(n) < 0.20),
        (age > 45) & (np.random.rand(n) < 0.32),
    ).astype(int)

    # BRCA mutation known
    brca_mutation = np.where(
        malignant == 0,
        np.random.rand(n) < 0.04,
        np.random.rand(n) < 0.12,
    ).astype(int)

    # Dense breast tissue (BI-RADS 3-4)
    dense_breast = np.where(
        malignant == 0,
        np.random.rand(n) < 0.30,
        np.random.rand(n) < 0.48,
    ).astype(int)

    # Palpable lump
    palpable_lump = np.where(
        malignant == 0,
        np.random.rand(n) < 0.20,
        np.random.rand(n) < 0.68,
    ).astype(int)

    # Nipple discharge
    nipple_discharge = np.where(
        malignant == 0,
        np.random.rand(n) < 0.08,
        np.random.rand(n) < 0.25,
    ).astype(int)

    # Skin changes
    skin_changes = np.where(
        malignant == 0,
        np.random.rand(n) < 0.04,
        np.random.rand(n) < 0.20,
    ).astype(int)

    # BMI (obesity increases post-menopausal risk)
    bmi = np.where(
        malignant == 0,
        np.random.normal(26.5, 5.5, n),
        np.random.normal(28.2, 5.8, n)
    ).clip(16, 55)

    # Alcohol use (drinks/week; >7 increases risk ~10%)
    alcohol_drinks_week = np.random.exponential(3.0, n).clip(0, 30).round(1)

    df = pd.DataFrame({
        # Nucleus measurements
        "radius_mean":          radius_mean.round(3),
        "texture_mean":         texture_mean.round(3),
        "perimeter_mean":       perimeter_mean.round(2),
        "area_mean":            area_mean.round(1),
        "smoothness_mean":      smoothness_mean.round(5),
        "compactness_mean":     compactness_mean.round(5),
        "concavity_mean":       concavity_mean.round(5),
        "concave_points_mean":  concave_points_mean.round(5),
        "symmetry_mean":        symmetry_mean.round(5),
        "fractal_dim_mean":     fractal_dim_mean.round(5),
        # Worst measurements
        "radius_worst":         radius_worst.round(3),
        "texture_worst":        texture_worst.round(3),
        "area_worst":           area_worst.round(1),
        "concavity_worst":      concavity_worst.round(5),
        "concave_pts_worst":    concave_pts_worst.round(5),
        # Clinical factors
        "age":                  age.round(1),
        "bmi":                  bmi.round(1),
        "alcohol_drinks_week":  alcohol_drinks_week,
        "family_history_bc":    family_history_bc,
        "prior_biopsy":         prior_biopsy,
        "hrt_use":              hrt_use,
        "brca_mutation":        brca_mutation,
        "dense_breast":         dense_breast,
        "palpable_lump":        palpable_lump,
        "nipple_discharge":     nipple_discharge,
        "skin_changes":         skin_changes,
        # Label
        "malignant":            malignant,
    })

    return df


if __name__ == "__main__":
    import pathlib
    pathlib.Path("data").mkdir(exist_ok=True)
    df = generate_breast_cancer_dataset()
    df.to_csv("data/breast_cancer_synthetic.csv", index=False)
    print(f"Dataset: {len(df)} rows | Malignant rate: {df.malignant.mean():.1%}")
    print(df.describe().T[["mean", "std", "min", "max"]])
