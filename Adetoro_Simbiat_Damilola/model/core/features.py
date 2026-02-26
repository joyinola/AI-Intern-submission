"""
Check Me — Feature Definitions
================================
Canonical ordered list of model input features and their human-readable
descriptions. Used by both the trainer and the inference/decision layer.
"""

FEATURE_COLS: list[str] = [
    # FNA nucleus measurements — mean values
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dim_mean",
    # FNA nucleus measurements — worst (largest) values
    "radius_worst",
    "texture_worst",
    "area_worst",
    "concavity_worst",
    "concave_pts_worst",
    # Clinical & demographic risk factors
    "age",
    "bmi",
    "alcohol_drinks_week",
    "family_history_bc",
    "prior_biopsy",
    "hrt_use",
    "brca_mutation",
    "dense_breast",
    "palpable_lump",
    "nipple_discharge",
    "skin_changes",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "radius_mean":          "Mean cell nucleus radius (FNA)",
    "texture_mean":         "Mean nucleus texture (grey-scale SD)",
    "perimeter_mean":       "Mean nucleus perimeter",
    "area_mean":            "Mean nucleus area (µm²)",
    "smoothness_mean":      "Nucleus contour smoothness",
    "compactness_mean":     "Nucleus compactness (perimeter²/area)",
    "concavity_mean":       "Severity of concave nucleus portions",
    "concave_points_mean":  "Number of concave nucleus points",
    "symmetry_mean":        "Nucleus symmetry",
    "fractal_dim_mean":     "Fractal dimension (coastline approx.)",
    "radius_worst":         "Largest nucleus radius (worst case)",
    "texture_worst":        "Worst-case nucleus texture",
    "area_worst":           "Largest nucleus area (worst case, µm²)",
    "concavity_worst":      "Worst-case nucleus concavity",
    "concave_pts_worst":    "Worst-case concave nucleus points",
    "age":                  "Patient age (years)",
    "bmi":                  "Body mass index (kg/m²)",
    "alcohol_drinks_week":  "Alcohol units per week",
    "family_history_bc":    "First-degree relative with breast cancer",
    "prior_biopsy":         "Previous breast biopsy",
    "hrt_use":              "Current hormone replacement therapy",
    "brca_mutation":        "Known BRCA1/2 pathogenic mutation",
    "dense_breast":         "Dense breast tissue (BI-RADS C/D)",
    "palpable_lump":        "Palpable breast lump on exam",
    "nipple_discharge":     "Spontaneous nipple discharge",
    "skin_changes":         "Skin dimpling / peau d'orange",
}
