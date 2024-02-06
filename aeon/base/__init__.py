"""Base classes for defining estimators and other objects in aeon."""

__author__ = ["mloning", "RNKuhns", "fkiraly", "TonyBagnall"]
__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseCollectionEstimator",
    "BaseSeriesEstimator",
    "_HeterogenousMetaEstimator",
    "load",
]

from aeon.base._base import BaseEstimator, BaseObject
from aeon.base._base_collection import BaseCollectionEstimator
from aeon.base._base_series import BaseSeriesEstimator
from aeon.base._meta import _HeterogenousMetaEstimator
from aeon.base._serialize import load
