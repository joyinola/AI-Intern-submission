"""
Check Me — Model Persistence
==============================
Handles saving and loading the trained sklearn pipeline and metrics JSON.
The model artefacts are written to a configurable directory (default: /model/artefacts).
"""

import json
import pickle
from pathlib import Path


_DEFAULT_PATH = Path("/model/artefacts")


def save_model(pipeline, metrics: dict, path: str | Path = _DEFAULT_PATH) -> None:
    """Persist the trained pipeline and metrics to disk."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

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
    pipeline_path = src / "pipeline.pkl"
    metrics_path  = src / "metrics.json"

    if not pipeline_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Model artefacts not found at '{src}'. "
            "Run 'python model/train.py' to train the model first."
        )

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    with open(metrics_path) as f:
        metrics = json.load(f)

    return pipeline, metrics
