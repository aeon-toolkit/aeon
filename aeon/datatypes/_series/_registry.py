"""Registry of mtypes for Series scitype. See datatypes._registry for API."""

import pandas as pd

__all__ = [
    "TYPE_REGISTER_SERIES",
    "TYPE_LIST_SERIES",
    "MTYPE_SOFT_DEPS_SERIES",
]


TYPE_REGISTER_SERIES = [
    ("pd.Series", "Series", "pd.Series representation of a univariate series"),
    (
        "pd.DataFrame",
        "Series",
        "pd.DataFrame representation of a uni- or multivariate series",
    ),
    (
        "np.ndarray",
        "Series",
        "2D numpy.ndarray with rows=samples, cols=variables, index=integers",
    ),
    (
        "xr.DataArray",
        "Series",
        "xr.DataArray representation of a uni- or multivariate series",
    ),
    (
        "dask_series",
        "Series",
        "xdas representation of a uni- or multivariate series",
    ),
]

MTYPE_SOFT_DEPS_SERIES = {"xr.DataArray": "xarray", "dask_series": "dask"}

TYPE_LIST_SERIES = pd.DataFrame(TYPE_REGISTER_SERIES)[0].values
