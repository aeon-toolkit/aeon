# -*- coding: utf-8 -*-
"""Panel transformations."""
__all__ = [
    "Catch22",
    "Catch22Wrapper",
    "ElbowClassSum",
    "ElbowClassPairwise",
    "DWTTransformer",
    "HOG1DTransformer",
    "TSInterpolator",
    "MatrixProfile",
    "PaddingTransformer",
    "RandomIntervals",
    "Tabularizer",
    "TimeBinner",
    "IntervalSegmenter",
    "SlidingWindowSegmenter",
    "RandomIntervalSegmenter",
    "RandomShapeletTransform",
    "RandomDilatedShapeletTransform",
    "ShapeletTransform",
    "SlopeTransformer",
    "SupervisedIntervals",
    "TruncationTransformer",
    "TSFreshFeatureExtractor",
    "TSFreshRelevantFeatureExtractor",
]

from aeon.transformations.collection.catch22 import Catch22
from aeon.transformations.collection.catch22wrapper import Catch22Wrapper
from aeon.transformations.collection.channel_selection import (
    ElbowClassPairwise,
    ElbowClassSum,
)
from aeon.transformations.collection.dilated_shapelet_transform import (
    RandomDilatedShapeletTransform,
)
from aeon.transformations.collection.dwt import DWTTransformer
from aeon.transformations.collection.hog1d import HOG1DTransformer
from aeon.transformations.collection.interpolate import TSInterpolator
from aeon.transformations.collection.matrix_profile import MatrixProfile
from aeon.transformations.collection.pad import PaddingTransformer
from aeon.transformations.collection.random_intervals import RandomIntervals
from aeon.transformations.collection.reduce import Tabularizer, TimeBinner
from aeon.transformations.collection.segment import (
    IntervalSegmenter,
    RandomIntervalSegmenter,
    SlidingWindowSegmenter,
)
from aeon.transformations.collection.shapelet_transform import (
    RandomShapeletTransform,
    ShapeletTransform,
)
from aeon.transformations.collection.slope import SlopeTransformer
from aeon.transformations.collection.supervised_intervals import SupervisedIntervals
from aeon.transformations.collection.truncate import TruncationTransformer
