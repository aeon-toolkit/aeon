# -*- coding: utf-8 -*-
"""Base class and type for numba distances."""
__author__ = ["chrisholder", "TonyBagnall"]
__all__ = [
    "NumbaDistance",
    "DistanceCallable",
    "DistanceFactoryCallable",
    "DistancePairwiseCallable",
    "ValidCallableTypes",
    "MetricInfo",
    "DistanceAlignmentPathCallable",
    "AlignmentPathReturn",
]

from aeon.distances.base._base import MetricInfo, NumbaDistance
from aeon.distances.base._types import (
    AlignmentPathReturn,
    DistanceAlignmentPathCallable,
    DistanceCallable,
    DistanceFactoryCallable,
    DistancePairwiseCallable,
    ValidCallableTypes,
)
