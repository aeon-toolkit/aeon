# -*- coding: utf-8 -*-
"""Ordinal time series classifiers."""
__all__ = [
  "OrdinalTemporalDictionaryEnsemble",
  "IndividualOrdinalTDE",
  "histogram_intersection",
]

from aeon.classification.ordinal_classification._ordinal_tde import (
    IndividualOrdinalTDE,
    OrdinalTemporalDictionaryEnsemble,
    histogram_intersection,
)
