"""
Check Me — Training Entry Point
================================
Run this script to generate the synthetic dataset, train the model,
and save artefacts to /model/artefacts/.

Usage:
    python model/train.py
    python model/train.py --n 5000           # larger dataset
    python model/train.py --out /custom/path # custom artefact path
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root or from model/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.dataset import generate_breast_cancer_dataset
from model.core.trainer import train_model
from model.core.persistence import save_model


def main(n: int = 3000, out: str = "/model/artefacts") -> None:
    print("=" * 55)
    print("  CHECK ME — Model Training")
    print("=" * 55)

    # 1. Dataset
    print(f"\n[1/3] Generating synthetic breast cancer dataset (n={n})...")
    df = generate_breast_cancer_dataset(n=n)
    data_path = Path(__file__).parent / "data" / "breast_cancer_synthetic.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"      Saved to {data_path}")
    print(f"      Rows: {len(df)} | Malignant rate: {df['malignant'].mean():.1%}")

    # 2. Train
    print("\n[2/3] Training Logistic Regression pipeline (5-fold CV)...")
    result   = train_model(df)
    pipeline = result["pipeline"]
    metrics  = result["metrics"]

    # 3. Report
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS (5-Fold Stratified CV)")
    print("=" * 55)
    print(f"  ROC-AUC   : {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    print(f"  F1        : {metrics['cv_f1_mean']:.3f}")
    print(f"  Recall    : {metrics['cv_recall_mean']:.3f}  ← minimising missed malignancies")
    print(f"  Precision : {metrics['cv_precision_mean']:.3f}")
    print(f"  Brier     : {metrics['brier_score']:.4f}  (calibration; lower = better)")

    print(f"\n  TOP FEATURE IMPORTANCES (Permutation AUC Drop)")
    print("  " + "-" * 45)
    for feat, imp in list(metrics["feature_importance"].items())[:10]:
        bar = "█" * max(1, int(imp * 600))
        print(f"  {feat:<28}  {imp:+.5f}  {bar}")

    # 4. Save
    print(f"\n[3/3] Saving artefacts to {out}...")
    save_model(pipeline, metrics, path=out)
    print("\n✓ Training complete. Start the API with: uvicorn api.main:app --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Check Me risk model")
    parser.add_argument("--n",   type=int, default=3000,             help="Dataset size")
    parser.add_argument("--out", type=str, default="/model/artefacts", help="Artefact output path")
    args = parser.parse_args()
    main(n=args.n, out=args.out)
