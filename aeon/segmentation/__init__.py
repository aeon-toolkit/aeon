"""Time Series Segmentation."""
__all__ = [
    "ClaSPSegmentation",
    "find_dominant_window_sizes",
    "GGS",
    "GreedyGaussianSegmentation",
    "IGTS",
    "InformationGainSegmentation",
    "entropy",
]

from aeon.segmentation._clasp import ClaSPSegmentation, find_dominant_window_sizes
from aeon.segmentation._ggs import GGS, GreedyGaussianSegmentation
from aeon.segmentation._igts import IGTS, InformationGainSegmentation, entropy
