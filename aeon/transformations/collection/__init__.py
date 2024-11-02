"""Collection transformations."""

__all__ = [
    # base class and series wrapper
    "BaseCollectionTransformer",
    # transformers
    "AutocorrelationFunctionTransformer",
    "ARCoefficientTransformer",
    "DownsampleTransformer",
    "ElbowClassSum",
    "ElbowClassPairwise",
    "DWTTransformer",
    "HOG1DTransformer",
    "Resizer",
    "MatrixProfile",
    "MinMax",
    "Padder",
    "PeriodogramTransformer",
    "Tabularizer",
    "SlopeTransformer",
    "Standardizer",
    "Truncator",
    "Normalizer",
]

from aeon.transformations.collection._acf import AutocorrelationFunctionTransformer
from aeon.transformations.collection._ar_coefficient import ARCoefficientTransformer
from aeon.transformations.collection._downsample import DownsampleTransformer
from aeon.transformations.collection._dwt import DWTTransformer
from aeon.transformations.collection._hog1d import HOG1DTransformer
from aeon.transformations.collection._matrix_profile import MatrixProfile
from aeon.transformations.collection._pad import Padder
from aeon.transformations.collection._periodogram import PeriodogramTransformer
from aeon.transformations.collection._reduce import Tabularizer
from aeon.transformations.collection._rescale import MinMax, Normalizer, Standardizer
from aeon.transformations.collection._resize import Resizer
from aeon.transformations.collection._slope import SlopeTransformer
from aeon.transformations.collection._truncate import Truncator
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.channel_selection import (
    ElbowClassPairwise,
    ElbowClassSum,
)
