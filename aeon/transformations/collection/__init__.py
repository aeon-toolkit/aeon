# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Collection transformations."""

__all__ = [
    # base class and series wrapper
    "BaseCollectionTransformer",
    "CollectionToSeriesWrapper",
    # transformers
    "AutocorrelationFunctionTransformer",
    "ARCoefficientTransformer",
    "Catch22",
    "ElbowClassSum",
    "ElbowClassPairwise",
    "DWTTransformer",
    "HOG1DTransformer",
    "TSInterpolator",
    "MatrixProfile",
    "PaddingTransformer",
    "PeriodogramTransformer",
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
    "SevenNumberSummaryTransformer",
    "SupervisedIntervals",
    "TruncationTransformer",
    "TSFreshFeatureExtractor",
    "TSFreshRelevantFeatureExtractor",
]

from aeon.transformations.collection._collection_wrapper import (
    CollectionToSeriesWrapper,
)
from aeon.transformations.collection.acf import AutocorrelationFunctionTransformer
from aeon.transformations.collection.ar_coefficient import ARCoefficientTransformer
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.catch22 import Catch22
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
from aeon.transformations.collection.periodogram import PeriodogramTransformer
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
from aeon.transformations.collection.summary import SevenNumberSummaryTransformer
from aeon.transformations.collection.supervised_intervals import SupervisedIntervals
from aeon.transformations.collection.truncate import TruncationTransformer
from aeon.transformations.collection.tsfresh import (
    TSFreshFeatureExtractor,
    TSFreshRelevantFeatureExtractor,
)
