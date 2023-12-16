"""Time Series Segmentation."""
__all__ = [
    "BaseSegmenter",
    "ClaSPSegmenter",
    "find_dominant_window_sizes",
    "GGS",
    "GreedyGaussianSegmenter",
    "IGTS",
    "InformationGainSegmenter",
    "entropy",
    "DummySegmenter",
    "EAggloSegmenter",
    "HMMSegmenter",
    "HidalgoSegmenter",
]

from aeon.segmentation._clasp import ClaSPSegmenter, find_dominant_window_sizes
from aeon.segmentation._dummy import DummySegmenter
from aeon.segmentation._eagglo import EAggloSegmenter
from aeon.segmentation._ggs import GGS, GreedyGaussianSegmenter
from aeon.segmentation._hidalgo import HidalgoSegmenter
from aeon.segmentation._hmm import HMMSegmenter
from aeon.segmentation._igts import IGTS, InformationGainSegmenter, entropy
from aeon.segmentation.base import BaseSegmenter
