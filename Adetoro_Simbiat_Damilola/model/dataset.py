
import pathlib

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Canonical path for the saved CSV
_DEFAULT_CSV = pathlib.Path(__file__).parent / "data" / "uci_breast_cancer.csv"


def load_uci_base() -> pd.DataFrame:
    """
    Load raw UCI dataset from sklearn (avoids network dependency).
    Returns DataFrame with underscore column names + 'malignant' label.
    sklearn encodes: 0 = malignant, 1 = benign → we invert so 1 = malignant.
    """
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=[c.replace(" ", "_") for c in bc.feature_names])
    df["malignant"] = (bc.target == 0).astype(int)   # 0 = malignant in sklearn
    return df


def augment_with_clinical_features(df: pd.DataFrame, seed: int = 2024) -> pd.DataFrame:
    """
    Augment the UCI FNA dataset with synthetic clinical risk factors.
    All distributions are class-conditional and drawn from published data.

    Columns added:
      age, bmi, alcohol_drinks_week, family_history_bc, prior_biopsy,
      hrt_use, brca_mutation, dense_breast, palpable_lump,
      nipple_discharge, skin_changes
    """
    rng = np.random.default_rng(seed)
    n   = len(df)
    mal = df["malignant"].values   # 1 = malignant

    def coin(p_benign: float, p_malignant: float) -> np.ndarray:
        """Binary feature with class-conditional probability."""
        p = np.where(mal == 1, p_malignant, p_benign)
        return (rng.random(n) < p).astype(int)

    # Age: malignant cases skew older (ACS: median dx age ~62)
    age = np.where(
        mal == 1,
        rng.normal(57, 13, n),
        rng.normal(47, 14, n),
    ).clip(25, 90).round(1)

    # BMI: post-menopausal obesity raises oestrogen levels
    bmi = np.where(
        mal == 1,
        rng.normal(28.2, 5.8, n),
        rng.normal(26.5, 5.5, n),
    ).clip(16, 55).round(1)

    # Alcohol: IARC: each 10g/day raises relative risk ~7-10%
    alcohol_drinks_week = rng.exponential(3.0, n).clip(0, 30).round(1)

    # Family history: doubles baseline risk (ACS 2023)
    family_history_bc = coin(0.12, 0.28)

    # Prior biopsy: atypical hyperplasia on biopsy raises risk 4×
    prior_biopsy = coin(0.15, 0.30)

    # HRT use: combined E+P HRT raises risk ~20% (WHI trial)
    hrt_use = ((age > 45) & (rng.random(n) < np.where(mal == 1, 0.32, 0.20))).astype(int)

    # BRCA1/2: lifetime risk 55–72% (BRCA1), 45–69% (BRCA2)
    brca_mutation = coin(0.04, 0.12)

    # Dense breast tissue (BI-RADS C/D): masks lesions, raises risk
    dense_breast = coin(0.30, 0.48)

    # Palpable lump: strongest clinical symptom
    palpable_lump = coin(0.20, 0.68)

    # Nipple discharge: non-lactational, unilateral = red flag
    nipple_discharge = coin(0.08, 0.25)

    # Skin changes: peau d'orange, dimpling — classic malignancy sign
    skin_changes = coin(0.04, 0.20)

    df = df.copy()
    df["age"]                = age
    df["bmi"]                = bmi
    df["alcohol_drinks_week"]= alcohol_drinks_week
    df["family_history_bc"]  = family_history_bc
    df["prior_biopsy"]       = prior_biopsy
    df["hrt_use"]            = hrt_use
    df["brca_mutation"]      = brca_mutation
    df["dense_breast"]       = dense_breast
    df["palpable_lump"]      = palpable_lump
    df["nipple_discharge"]   = nipple_discharge
    df["skin_changes"]       = skin_changes

    return df


def load_dataset(csv_path: str | pathlib.Path = _DEFAULT_CSV) -> pd.DataFrame:
    """
    Load the augmented UCI dataset. If the CSV doesn't exist, build it from
    sklearn's embedded copy of the UCI data and save it.

    Returns:
        pd.DataFrame with all FEATURE_COLS + 'malignant' column.
    """
    csv_path = pathlib.Path(csv_path)

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = load_uci_base()
        df = augment_with_clinical_features(df)
        df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Malignant rate: {df.malignant.mean():.1%}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nSample:\n{df.head(3).to_string()}")
