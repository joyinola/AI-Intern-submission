"""
Check Me — Model Persistence
==============================
Handles saving and loading the trained sklearn pipeline and metrics JSON.
The model artefacts are written to a configurable directory (default: /model/artefacts).

Uses joblib for pipeline serialisation — joblib is the recommended format for
sklearn pipelines because it uses efficient numpy array pickling (mmap-friendly)
and is significantly faster than plain pickle for large numpy arrays embedded in
trained tree ensembles like GradientBoostingClassifier.
"""

import json
import joblib
from pathlib import Path


_DEFAULT_PATH = Path("/model/artefacts")


def save_model(pipeline, metrics: dict, path: str | Path = _DEFAULT_PATH) -> None:
    """Persist the trained pipeline and metrics to disk."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    # Save pipeline as .joblib (recommended for sklearn models)
    joblib.dump(pipeline, out / "pipeline.joblib")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Model saved to {out}/")


def load_model(path: str | Path = _DEFAULT_PATH) -> tuple:
    """
    Load pipeline and metrics from disk.

    Returns:
        (pipeline, metrics) tuple.

    Raises:
        FileNotFoundError if artefacts are missing (model not yet trained).
    """
    src = Path(path)
    pipeline_path = src / "pipeline.joblib"
    metrics_path  = src / "metrics.json"

    if not pipeline_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Model artefacts not found at '{src}'. "
            "Run 'python model/train.py' to train the model first."
        )

    pipeline = joblib.load(pipeline_path)

    with open(metrics_path) as f:
        metrics = json.load(f)

    return pipeline, metrics
