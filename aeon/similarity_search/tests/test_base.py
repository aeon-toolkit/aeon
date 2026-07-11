"""Test for similarity search base class."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest

from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockSubsequenceSearch,
    MockWholeSeriesSearch,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT, _get_datatypes_for_estimator


def test_input_shape_fit_predict():
    """Test input shapes for similarity search.

    Fit takes a collection (3D), predict takes a single series (2D). This combined
    version covers both the subsequence and whole-series mock estimators, so the
    per-subpackage base tests do not need to repeat it.
    """
    estimator_ss = MockSubsequenceSearch()
    estimator_ws = MockWholeSeriesSearch()
    datatypes_ss = _get_datatypes_for_estimator(estimator_ss)
    datatypes_ws = _get_datatypes_for_estimator(estimator_ws)

    # dummy data to pass to fit when testing predict/predict_proba
    for datatype in datatypes_ss:
        X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
        X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
        # fit takes a collection, predict takes a single series
        estimator_ss.fit(X_train, y_train).predict(X_test[0])

    for datatype in datatypes_ws:
        X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
        X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
        # fit takes a collection, predict takes a single series
        estimator_ws.fit(X_train, y_train).predict(X_test[0])


def test_predict_channel_mismatch_raises():
    """Test predict raises a ValueError when channel counts differ from fit."""
    estimator = MockSubsequenceSearch(length=20)
    X_train = make_example_3d_numpy(
        n_cases=5, n_channels=2, n_timepoints=20, return_y=False
    )
    estimator.fit(X_train)

    X_test = make_example_2d_numpy_series(n_channels=1, n_timepoints=20)
    with pytest.raises(ValueError, match="Expected X to have 2 channels"):
        estimator.predict(X_test)


@pytest.mark.parametrize("k", [0, -1, 2.5, "3", np.nan])
def test_predict_invalid_k_raises(k):
    """Test predict raises a ValueError for k that is not a positive int or np.inf."""
    estimator = MockSubsequenceSearch(length=10)
    X_train = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=50, return_y=False
    )
    estimator.fit(X_train)
    query = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        estimator.predict(query, k=k)
