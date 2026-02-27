
import sys
from pathlib import Path
from functools import lru_cache

# Ensure project root is on the path when running inside Docker
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.core.persistence import load_model


@lru_cache(maxsize=1)
def get_model():
    """
    Load and cache the trained pipeline + metrics.
    Called once on first request; result is reused for the process lifetime.
    Raises FileNotFoundError if model artefacts are missing.
    """
    pipeline, metrics = load_model()
    return pipeline, metrics
