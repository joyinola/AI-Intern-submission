"""
Check Me — Feature Definitions
================================
Uses the real UCI Wisconsin Breast Cancer Diagnostic dataset column names,
augmented with clinical/demographic risk factors.

UCI dataset reference:
  Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995).
  UCI Machine Learning Repository.
  https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
"""

FEATURE_COLS: list[str] = [
    # FNA nucleus measurements — mean values (UCI real columns)
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    # FNA nucleus measurements — worst (largest) values
    "worst_radius",
    "worst_texture",
    "worst_area",
    "worst_concavity",
    "worst_concave_points",
    # Clinical & demographic risk factors (not in original UCI — appended synthetically
    # based on published literature distributions)
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
    "mean_radius":           "Mean cell nucleus radius (FNA)",
    "mean_texture":          "Mean nucleus texture (grey-scale SD)",
    "mean_perimeter":        "Mean nucleus perimeter",
    "mean_area":             "Mean nucleus area (µm²)",
    "mean_smoothness":       "Nucleus contour smoothness",
    "mean_compactness":      "Nucleus compactness (perimeter²/area)",
    "mean_concavity":        "Severity of concave nucleus portions",
    "mean_concave_points":   "Number of concave nucleus points",
    "mean_symmetry":         "Nucleus symmetry",
    "mean_fractal_dimension":"Fractal dimension (coastline approx.)",
    "worst_radius":          "Largest nucleus radius (worst case)",
    "worst_texture":         "Worst-case nucleus texture",
    "worst_area":            "Largest nucleus area (worst case, µm²)",
    "worst_concavity":       "Worst-case nucleus concavity",
    "worst_concave_points":  "Worst-case concave nucleus points",
    "age":                   "Patient age (years)",
    "bmi":                   "Body mass index (kg/m²)",
    "alcohol_drinks_week":   "Alcohol units per week",
    "family_history_bc":     "First-degree relative with breast cancer",
    "prior_biopsy":          "Previous breast biopsy",
    "hrt_use":               "Current hormone replacement therapy",
    "brca_mutation":         "Known BRCA1/2 pathogenic mutation",
    "dense_breast":          "Dense breast tissue (BI-RADS C/D)",
    "palpable_lump":         "Palpable breast lump on exam",
    "nipple_discharge":      "Spontaneous nipple discharge",
    "skin_changes":          "Skin dimpling / peau d'orange",
}
