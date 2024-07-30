"""Base classes for defining estimators and other objects in aeon."""

__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseCollectionEstimator",
    "BaseSeriesEstimator",
    "_HeterogenousMetaEstimator",
]

import os
import warnings

import aeon
from aeon.base._base import BaseEstimator, BaseObject
from aeon.base._base_collection import BaseCollectionEstimator
from aeon.base._base_series import BaseSeriesEstimator
from aeon.base._meta import _HeterogenousMetaEstimator

if (
    aeon.AEON_DEPRECATION_WARNING
    and os.environ.get("AEON_DEPRECATION_WARNING", "true").lower() != "false"
):
    warnings.warn(
        "The aeon package will soon be releasing v1.0.0 with the removal of "
        "legacy modules and interfaces such as BaseTransformer and BaseForecaster. "
        "This will contain breaking changes. See aeon-toolkit.org for more "
        "information. Set aeon.AEON_DEPRECATION_WARNING or the "
        "AEON_DEPRECATION_WARNING environmental variable to 'False' to disable this "
        "warning.",
        FutureWarning,
        stacklevel=0,
    )
    aeon.AEON_DEPRECATION_WARNING = False
