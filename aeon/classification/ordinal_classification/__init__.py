"""Ordinal time series classifiers."""

__all__ = [
    "OrdinalTDE",
    "IndividualOrdinalTDE",
    "histogram_intersection",
]

from aeon.classification.ordinal_classification._ordinal_tde import (
    IndividualOrdinalTDE,
    OrdinalTDE,
    histogram_intersection,
)
