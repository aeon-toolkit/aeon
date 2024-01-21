"""Collection transformations."""

__all__ = [
    # base class and series wrapper
    "BaseCollectionTransformer",
    "CollectionToSeriesWrapper",
    # transformers
    "AutocorrelationFunctionTransformer",
    "ARCoefficientTransformer",
    "ElbowClassSum",
    "ElbowClassPairwise",
    "DWTTransformer",
    "HOG1DTransformer",
    "TSInterpolator",
    "MatrixProfile",
    "PaddingTransformer",
    "PeriodogramTransformer",
    "Tabularizer",
    "IntervalSegmenter",
    "RandomIntervalSegmenter",
    "SlidingWindowSegmenter",
    "SlopeTransformer",
    "TimeSeriesScaler",
    "TruncationTransformer",
]

from aeon.transformations.collection._collection_wrapper import (
    CollectionToSeriesWrapper,
)
from aeon.transformations.collection.acf import AutocorrelationFunctionTransformer
from aeon.transformations.collection.ar_coefficient import ARCoefficientTransformer
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.channel_selection import (
    ElbowClassPairwise,
    ElbowClassSum,
)
from aeon.transformations.collection.dwt import DWTTransformer
from aeon.transformations.collection.hog1d import HOG1DTransformer
from aeon.transformations.collection.interpolate import TSInterpolator
from aeon.transformations.collection.matrix_profile import MatrixProfile
from aeon.transformations.collection.pad import PaddingTransformer
from aeon.transformations.collection.periodogram import PeriodogramTransformer
from aeon.transformations.collection.reduce import Tabularizer
from aeon.transformations.collection.scaler import TimeSeriesScaler
from aeon.transformations.collection.segment import (
    IntervalSegmenter,
    RandomIntervalSegmenter,
    SlidingWindowSegmenter,
)
from aeon.transformations.collection.slope import SlopeTransformer
from aeon.transformations.collection.truncate import TruncationTransformer
