"""Time Series Segmentation."""

__all__ = [
    "BaseSegmenter",
    "ClaSPSegmenter",
    "find_dominant_window_sizes",
    "GreedyGaussianSegmenter",
    "InformationGainSegmenter",
    "entropy",
    "RandomSegmenter",
    "EAggloSegmenter",
    "HMMSegmenter",
    "HidalgoSegmenter",
]

from aeon.segmentation._clasp import ClaSPSegmenter, find_dominant_window_sizes
from aeon.segmentation._eagglo import EAggloSegmenter
from aeon.segmentation._ggs import GreedyGaussianSegmenter
from aeon.segmentation._hidalgo import HidalgoSegmenter
from aeon.segmentation._hmm import HMMSegmenter
from aeon.segmentation._igts import InformationGainSegmenter, entropy
from aeon.segmentation._random import RandomSegmenter
from aeon.segmentation.base import BaseSegmenter
