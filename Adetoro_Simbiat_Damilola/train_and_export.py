"""
Check Me — Local Training & Export Script
==========================================
Run this script LOCALLY (not in Docker) to retrain the model and save
the artefacts into model/artefacts/. Then rebuild the Docker image to
bundle the new artefacts.

Workflow:
    1. python train_and_export.py          # retrain & save artefacts
    2. docker compose up --build           # bundle artefacts into image
    3. docker compose up                   # starts in seconds every time after

The Docker container NEVER trains — it only loads the pre-saved artefacts.
This means docker compose up takes < 10 seconds regardless of model complexity.

Usage:
    python train_and_export.py
    python train_and_export.py --out model/artefacts   # custom output path
"""

import argparse
import json
import pathlib
import sys
import time

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from model.dataset import load_dataset
from model.core.trainer import train_model
from model.core.persistence import save_model


def main(out_dir: str = "model/artefacts") -> None:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Check Me — Local Model Training & Export")
    print("=" * 60)
    print(f"\nOutput directory: {out_path.resolve()}")
    print("\nThis runs LOCALLY. The trained artefacts are committed to the")
    print("repo and bundled into Docker — no training happens in the container.\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("[1/3] Loading augmented UCI breast cancer dataset...")
    df = load_dataset()
    print(f"      {len(df)} patients | {df.malignant.mean():.1%} malignant rate")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Training GradientBoosting pipeline...")
    t0 = time.time()
    result   = train_model(df)
    pipeline = result["pipeline"]
    metrics  = result["metrics"]
    elapsed  = time.time() - t0

    print(f"      Done in {elapsed:.1f}s")
    print(f"\n      ┌─────────────────────────────────┐")
    print(f"      │  ROC-AUC  {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}      │")
    print(f"      │  Recall   {metrics['cv_recall_mean']:.4f}                │")
    print(f"      │  F1       {metrics['cv_f1_mean']:.4f}                │")
    print(f"      │  Brier    {metrics['brier_score']:.4f}                │")
    print(f"      └─────────────────────────────────┘")
    print(f"\n      Top 5 features (permutation importance):")
    for feat, imp in list(metrics["feature_importance"].items())[:5]:
        bar = "█" * max(1, int(imp * 300))
        print(f"        {feat:<30} {imp:.5f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving artefacts to {out_path}/...")
    save_model(pipeline, metrics, str(out_path))

    # Write clean metrics.json (no raw feature_importance dict — use top_features list)
    metrics_export = {k: v for k, v in metrics.items() if k != "feature_importance"}
    metrics_export["top_features"] = list(metrics["feature_importance"].items())[:15]
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)

    print(f"      pipeline.joblib  ({(out_path / 'pipeline.joblib').stat().st_size / 1024:.0f} KB)")
    print(f"      metrics.json")

    print(f"""
✓ Artefacts saved. Next steps:

  Rebuild Docker image to bundle the new artefacts:
    docker compose up --build

  Or just restart if Docker image is already built:
    docker compose up

  The container will start in < 10 seconds — no training in Docker.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Check Me model and export artefacts.")
    parser.add_argument("--out", default="model/artefacts", help="Output directory for artefacts")
    args = parser.parse_args()
    main(args.out)
