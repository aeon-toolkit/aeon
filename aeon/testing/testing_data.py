"""Datasets used for testing."""

import numpy as np

from aeon.base import BaseCollectionEstimator, BaseSeriesEstimator
from aeon.testing.data_generation import (
    make_example_2d_dataframe,
    make_example_2d_numpy,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_list,
    make_example_multi_index_dataframe,
    make_example_nested_dataframe,
)

data_rng = np.random.RandomState(42)

X_collection, y_collection = make_example_3d_numpy(
    n_cases=10,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_collection2, y_collection2 = make_example_3d_numpy(
    n_cases=5,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
y_collection_r = y_collection.astype(np.float32) + data_rng.uniform(
    size=y_collection.shape
)
y_collection2_r = y_collection2.astype(np.float32) + data_rng.uniform(
    size=y_collection2.shape
)

X_collection_mv, y_collection_mv = make_example_3d_numpy(
    n_cases=10,
    n_channels=2,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_collection_mv2, y_collection_mv2 = make_example_3d_numpy(
    n_cases=5,
    n_channels=2,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
y_collection_mv_r = y_collection.astype(np.float32) + data_rng.uniform(
    size=y_collection.shape
)
y_collection_mv2_r = y_collection2.astype(np.float32) + data_rng.uniform(
    size=y_collection2.shape
)

X_collection_ul, y_collection_ul = make_example_3d_numpy_list(
    n_cases=10,
    n_channels=1,
    min_n_timepoints=10,
    max_n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_collection_ul2, y_collection_ul2 = make_example_3d_numpy_list(
    n_cases=5,
    n_channels=1,
    min_n_timepoints=10,
    max_n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
y_collection_ul_r = y_collection.astype(np.float32) + data_rng.uniform(
    size=y_collection.shape
)
y_collection_ul2_r = y_collection2.astype(np.float32) + data_rng.uniform(
    size=y_collection2.shape
)

X_collection_mi, y_collection_mi = make_example_3d_numpy(
    n_cases=10,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
X_collection_mi2, y_collection_mi2 = make_example_3d_numpy(
    n_cases=5,
    n_channels=1,
    n_timepoints=20,
    random_state=data_rng.randint(np.iinfo(np.int32).max),
)
y_collection_mi_r = y_collection.astype(np.float32) + data_rng.uniform(
    size=y_collection.shape
)
y_collection_mi2_r = y_collection2.astype(np.float32) + data_rng.uniform(
    size=y_collection2.shape
)


TEST_DATA_DICT = {
    "UnivariateCollection": {"train": X_collection, "test": X_collection2},
    "MultivariateCollection": {"train": X_collection_mv, "test": X_collection_mv2},
    "UnequalLengthCollection": {"train": X_collection_ul, "test": X_collection_ul2},
    "MissingValuesCollection": {"train": X_collection_mi, "test": X_collection_mi2},
}
TEST_LABEL_DICT = {
    "Classification": {
        "train": y_collection,
        "test": y_collection2,
    },
    "Regression": {
        "train": y_collection_r,
        "test": y_collection2_r,
    },
    "UnivariateCollectionClassification": {
        "train": y_collection,
        "test": y_collection2,
    },
    "UnivariateCollectionRegression": {
        "train": y_collection_r,
        "test": y_collection2_r,
    },
    "MultivariateCollectionClassification": {
        "train": y_collection_mv,
        "test": y_collection_mv2,
    },
    "MultivariateCollectionRegression": {
        "train": y_collection_mv_r,
        "test": y_collection_mv2_r,
    },
    "UnequalLengthCollectionClassification": {
        "train": y_collection_ul,
        "test": y_collection_ul2,
    },
    "UnequalLengthCollectionRegression": {
        "train": y_collection_ul_r,
        "test": y_collection_ul2_r,
    },
    "MissingValuesCollectionClassification": {
        "train": y_collection_mi,
        "test": y_collection_mi2,
    },
    "MissingValuesCollectionRegression": {
        "train": y_collection_mi_r,
        "test": y_collection_mi2_r,
    },
}

EQUAL_LENGTH_UNIVARIATE = {
    "numpy3D": X_collection,
    "numpy2D": make_example_2d_numpy(
        n_cases=10,
        n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "np-list": make_example_3d_numpy_list(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "df-list": make_example_dataframe_list(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "pd-wide": make_example_2d_dataframe(
        n_cases=10,
        n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "nested_univ": make_example_nested_dataframe(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "pd-multiindex": make_example_multi_index_dataframe(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
}

UNEQUAL_LENGTH_UNIVARIATE = {
    "np-list": X_collection_ul,
    "df-list": make_example_dataframe_list(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=10,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "nested_univ": make_example_nested_dataframe(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=10,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "pd-multiindex": make_example_multi_index_dataframe(
        n_cases=10,
        n_channels=1,
        min_n_timepoints=10,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
}


EQUAL_LENGTH_MULTIVARIATE = {
    "numpy3D": X_collection_mv,
    "np-list": make_example_3d_numpy_list(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "df-list": make_example_dataframe_list(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "nested_univ": make_example_nested_dataframe(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "pd-multiindex": make_example_multi_index_dataframe(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=20,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
}

UNEQUAL_LENGTH_MULTIVARIATE = {
    "np-list": make_example_3d_numpy_list(
        n_cases=10,
        n_channels=2,
        max_n_timepoints=20,
        min_n_timepoints=10,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "df-list": make_example_dataframe_list(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=10,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "nested_univ": make_example_nested_dataframe(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=10,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
    "pd-multiindex": make_example_multi_index_dataframe(
        n_cases=10,
        n_channels=2,
        min_n_timepoints=10,
        max_n_timepoints=20,
        random_state=data_rng.randint(np.iinfo(np.int32).max),
        return_y=False,
    ),
}


def get_data_types_for_estimator(estimator):
    """Get data types for estimator.

    Parameters
    ----------
    estimator : BaseEstimator instance or class
        Estimator instance or class to check for valid input data types.

    Returns
    -------
    datatypes : list of str
        List of valid data types for the estimator usable in TEST_DATA_DICT.
    """
    univariate = estimator.get_tag("capability:univariate", True, raise_error=False)
    multivariate = estimator.get_tag(
        "capability:multivariate", False, raise_error=False
    )
    unequal_length = estimator.get_tag(
        "capability:unequal_length", False, raise_error=False
    )
    missing_values = estimator.get_tag(
        "capability:missing_values", False, raise_error=False
    )
    datatypes = []

    if isinstance(estimator, BaseCollectionEstimator):
        if univariate:
            datatypes.append("UnivariateCollection")
        if multivariate:
            datatypes.append("MultivariateCollection")
        if unequal_length:
            datatypes.append("UnequalLengthCollection")
        if missing_values:
            datatypes.append("MissingValuesCollection")
    elif isinstance(estimator, BaseSeriesEstimator):
        if univariate:
            datatypes.append("UnivariateSeries")
        if multivariate:
            datatypes.append("MultivariateSeries")
        if missing_values:
            datatypes.append("MissingValuesSeries")
    else:
        raise ValueError(f"Unknown estimator type: {type(estimator)}")

    if len(datatypes) == 0:
        raise ValueError(f"No valid data types found for estimator {estimator}")

    return datatypes
