"""Registry of mtypes for Hierarchical scitype. See datatypes._registry for API."""

import pandas as pd

__all__ = [
    "TYPE_REGISTER_HIERARCHICAL",
    "TYPE_LIST_HIERARCHICAL",
    "TYPE_SOFT_DEPS_HIERARCHICAL",
]


TYPE_REGISTER_HIERARCHICAL = [
    (
        "pd_multiindex_hier",
        "Hierarchical",
        "pd.DataFrame with MultiIndex",
    ),
    (
        "dask_hierarchical",
        "Hierarchical",
        "dask frame with multiple hierarchical indices, as per dask_to_pd convention",
    ),
]

TYPE_SOFT_DEPS_HIERARCHICAL = {"dask_hierarchical": "dask"}

TYPE_LIST_HIERARCHICAL = pd.DataFrame(TYPE_REGISTER_HIERARCHICAL)[0].values
