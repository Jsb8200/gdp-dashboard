"""LightGBM magnitude detector.

Detects, *before* the move starts, bars where a high-confluence /
high-magnitude move is imminent. See MAGNITUDE_DETECTOR.md for usage.
"""

from .config import Config
from .data import load_ohlcv, make_synthetic_ohlcv
from .features import build_features
from .labels import build_labels
from .model import MagnitudeDetector

__all__ = [
    "Config",
    "load_ohlcv",
    "make_synthetic_ohlcv",
    "build_features",
    "build_labels",
    "MagnitudeDetector",
]
