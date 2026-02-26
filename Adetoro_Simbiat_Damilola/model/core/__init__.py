"""
Check Me — Model Core Package
Exposes the public interface for the hybrid breast cancer risk engine.
"""

from .flags import CLINICAL_FLAGS, ClinicalFlag, evaluate_clinical_flags
from .features import FEATURE_COLS, FEATURE_DESCRIPTIONS
from .trainer import train_model
from .decision import make_decision, CheckMeDecision
from .persistence import save_model, load_model

__all__ = [
    "CLINICAL_FLAGS",
    "ClinicalFlag",
    "evaluate_clinical_flags",
    "FEATURE_COLS",
    "FEATURE_DESCRIPTIONS",
    "train_model",
    "make_decision",
    "CheckMeDecision",
    "save_model",
    "load_model",
]
