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

ALL_TIME_SERIES_TYPES = [
    "pd-wide",
    "np-list",
    "dask_series",
    "dask_panel",
    "pd-long",
    "pd.Series",
    "numpy3D",
    "xr.DataArray",
    "pd.DataFrame",
    "dask_hierarchical",
    "pd_multiindex_hier",
    "df-list",
    "nested_univ",
    "pd-multiindex",
    "np.ndarray",
]
