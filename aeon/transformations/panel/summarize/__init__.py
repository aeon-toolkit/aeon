# -*- coding: utf-8 -*-
"""Module for summarization transformers."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
__all__ = [
    "DerivativeSlopeTransformer",
    "PlateauFinder",
    "RandomIntervalFeatureExtractor",
    "FittedParamExtractor",
    "series_slope_derivative",
]

from ._extract import (
    DerivativeSlopeTransformer,
    FittedParamExtractor,
    PlateauFinder,
    RandomIntervalFeatureExtractor,
    series_slope_derivative,
)
