"""
Check Me — Model Training Entry Point
======================================
Loads the augmented UCI dataset, trains the GradientBoosting model,
evaluates it, and saves artefacts.

Usage (both work):
    python -m model.train                        # from /app (recommended)
    python model/train.py                        # also works
"""

import argparse
import json
import pathlib
import sys

# ── Bootstrap: make relative imports work when run as `python model/train.py` ─
# When invoked directly Python sets __package__ to None, breaking relative
# imports. We detect this and re-launch via runpy as a proper package module.
if __name__ == "__main__" and __package__ is None:
    _project_root = str(pathlib.Path(__file__).parent.parent.resolve())
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    import runpy
    runpy.run_module("model.train", run_name="__main__", alter_sys=True)
    sys.exit(0)

# ── Package-relative imports (active once __package__ is set) ────────────────
from .dataset import load_dataset
from .core.trainer import train_model
from .core.persistence import save_model


def main(out_dir: str = "/model/artefacts") -> None:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Check Me — GradientBoosting Risk Model Training")
    print("=" * 60)

    print("\n[1/4] Loading augmented UCI breast cancer dataset...")
    df = load_dataset()
    print(f"      {len(df)} patients | {df.malignant.mean():.1%} malignant")

    print("\n[2/4] Training GradientBoosting pipeline (5-fold CV)...")
    result = train_model(df)
    pipeline = result["pipeline"]
    metrics  = result["metrics"]

    print("\n[3/4] Model Performance:")
    print(f"      Model:          {metrics['model_type']}")
    print(f"      ROC-AUC:        {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
    print(f"      F1 Score:       {metrics['cv_f1_mean']:.4f}")
    print(f"      Recall:         {metrics['cv_recall_mean']:.4f}")
    print(f"      Precision:      {metrics['cv_precision_mean']:.4f}")
    print(f"      Brier Score:    {metrics['brier_score']:.4f}")
    print(f"\n      Top 5 features by permutation importance:")
    for feat, imp in list(metrics["feature_importance"].items())[:5]:
        print(f"        {feat:<30} {imp:.5f}")

    print(f"\n[4/4] Saving artefacts to {out_path}...")
    save_model(pipeline, metrics, str(out_path))

    metrics_path = out_path / "metrics.json"
    json_metrics = {k: v for k, v in metrics.items() if k != "feature_importance"}
    json_metrics["top_features"] = list(metrics["feature_importance"].items())[:10]
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)

    print(f"      pipeline.pkl saved")
    print(f"      metrics.json saved")
    print("\n✓ Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/model/artefacts")
    args = parser.parse_args()
    main(args.out)
