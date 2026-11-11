"""
LifeMate CS-ASR: Closed-Loop Bilingual Semantic Regularization & Dynamic Thresholding
Sub-1.5B Conversational Vietnamese-English Code-Switching Speech Recognition
LifeMate CS-ASR Official Research Implementation
"""

from .config import CONFIG
from .cs_asr import (
    ClosedLoopASRPipeline,
    norm,
    norm_ed,
    calc_syer,
    calc_ewer,
    extract_features
)

__version__ = "1.0.0"
__all__ = [
    "CONFIG",
    "ClosedLoopASRPipeline",
    "norm",
    "norm_ed",
    "calc_syer",
    "calc_ewer",
    "extract_features"
]
