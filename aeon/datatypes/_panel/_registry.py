"""Registry of mtypes for Collections. See datatypes._registry for API."""

import pandas as pd

__all__ = [
    "TYPE_REGISTER_PANEL",
    "TYPE_LIST_PANEL",
    "TYPE_SOFT_DEPS_PANEL",
]


TYPE_REGISTER_PANEL = [
    (
        "nested_univ",
        "Panel",
        "pd.DataFrame with one column per channel, pd.Series in cells",
    ),
    (
        "numpy3D",
        "Panel",
        "3D np.ndarray of format (n_cases, n_channels, n_timepoints)",
    ),
    (
        "numpy2D",
        "Panel",
        "2D np.ndarray of format (n_cases, n_timepoints)",
    ),
    ("pd-multiindex", "Panel", "pd.DataFrame with multi-index (instances, timepoints)"),
    ("pd-wide", "Panel", "pd.DataFrame in wide format, cols = (instance*timepoints)"),
    (
        "pd-long",
        "Panel",
        "pd.DataFrame in long format, cols = (index, time_index, column)",
    ),
    ("df-list", "Panel", "list of pd.DataFrame"),
    (
        "dask_panel",
        "Panel",
        "dask frame with one instance and one time index, as per dask_to_pd convention",
    ),
    (
        "np-list",
        "Panel",
        "list of length [n_cases], each case a 2D np.ndarray of shape (n_channels, "
        "n_timepoints)",
    ),
]

TYPE_SOFT_DEPS_PANEL = {"xr.DataArray": "xarray", "dask_panel": "dask"}

TYPE_LIST_PANEL = pd.DataFrame(TYPE_REGISTER_PANEL)[0].values
