"""Tests for collection data generation functions."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pandas as pd
import pytest

from aeon.testing.data_generation import (
    make_example_2d_dataframe_collection,
    make_example_2d_numpy_collection,
    make_example_2d_numpy_list,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_list,
    make_example_multi_index_dataframe,
)
from aeon.utils.validation.collection import get_type

N_CASES = [5, 10]
N_CHANNELS = [1, 3]
MIN_N_TIMEPOINTS = [6, 8]
MAX_N_TIMEPOINTS = [8, 10]
N_CLASSES = [2, 5]
REGRESSION = [True, False]


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_3d_numpy(
    n_cases, n_channels, n_timepoints, n_classes, regression
):
    """Test generated numpy3d data is in the correct format."""
    X, y = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_labels=n_classes,
        regression_target=regression,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_cases, n_channels, n_timepoints)
    assert y.shape == (n_cases,)
    assert get_type(X) == "numpy3D"
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_2d_numpy(n_cases, n_timepoints, n_classes, regression):
    """Test generated numpy2d data is in the correct format."""
    X, y = make_example_2d_numpy_collection(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
        n_labels=n_classes,
        regression_target=regression,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_cases, n_timepoints)
    assert y.shape == (n_cases,)
    assert get_type(X) == "numpy2D"
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("min_n_timepoints", MIN_N_TIMEPOINTS)
@pytest.mark.parametrize("max_n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_3d_numpy_list(
    n_cases, n_channels, min_n_timepoints, max_n_timepoints, n_classes, regression
):
    """Test generated 3d numpy list data is in the correct format."""
    X, y = make_example_3d_numpy_list(
        n_cases=n_cases,
        n_channels=n_channels,
        n_labels=n_classes,
        min_n_timepoints=min_n_timepoints,
        max_n_timepoints=max_n_timepoints,
        regression_target=regression,
    )

    assert isinstance(X, list)
    assert all(isinstance(x, np.ndarray) for x in X)
    assert isinstance(y, np.ndarray)
    assert len(X) == n_cases
    assert all([x.shape[0] == n_channels for x in X])
    if min_n_timepoints == max_n_timepoints:
        assert all([x.shape[1] == min_n_timepoints for x in X])
    else:
        assert all(
            [
                x.shape[1] >= min_n_timepoints and x.shape[1] <= max_n_timepoints
                for x in X
            ]
        )
    assert y.shape == (n_cases,)
    assert get_type(X) == "np-list"
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("min_n_timepoints", MIN_N_TIMEPOINTS)
@pytest.mark.parametrize("max_n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_2d_numpy_list(
    n_cases, min_n_timepoints, max_n_timepoints, n_classes, regression
):
    """Test generated 2d numpy list data is in the correct format."""
    X, y = make_example_2d_numpy_list(
        n_cases=n_cases,
        min_n_timepoints=min_n_timepoints,
        max_n_timepoints=max_n_timepoints,
        n_labels=n_classes,
        regression_target=regression,
    )

    assert isinstance(X, list)
    assert all(isinstance(x, np.ndarray) for x in X)
    assert isinstance(y, np.ndarray)
    assert len(X) == n_cases
    if min_n_timepoints == max_n_timepoints:
        assert all([x.shape[0] == min_n_timepoints for x in X])
    else:
        assert all(
            [
                x.shape[0] >= min_n_timepoints and x.shape[0] <= max_n_timepoints
                for x in X
            ]
        )
    assert y.shape == (n_cases,)
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("min_n_timepoints", MIN_N_TIMEPOINTS)
@pytest.mark.parametrize("max_n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_dataframe_list(
    n_cases, n_channels, min_n_timepoints, max_n_timepoints, n_classes, regression
):
    """Test generated DataFrame list data is in the correct format."""
    X, y = make_example_dataframe_list(
        n_cases=n_cases,
        n_channels=n_channels,
        n_labels=n_classes,
        min_n_timepoints=min_n_timepoints,
        max_n_timepoints=max_n_timepoints,
        regression_target=regression,
    )

    assert isinstance(X, list)
    assert all(isinstance(x, pd.DataFrame) for x in X)
    assert isinstance(y, np.ndarray)
    assert len(X) == n_cases
    assert all([x.shape[0] == n_channels for x in X])
    if min_n_timepoints == max_n_timepoints:
        assert all([x.shape[1] == min_n_timepoints for x in X])
    else:
        assert all(
            [
                x.shape[1] >= min_n_timepoints and x.shape[1] <= max_n_timepoints
                for x in X
            ]
        )
    assert y.shape == (n_cases,)
    assert get_type(X) == "df-list"
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_2d_dataframe(n_cases, n_timepoints, n_classes, regression):
    """Test generated pd-wide data is in the correct format."""
    X, y = make_example_2d_dataframe_collection(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
        n_labels=n_classes,
        regression_target=regression,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_cases, n_timepoints)
    assert y.shape == (n_cases,)
    assert get_type(X) == "pd-wide"
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("min_n_timepoints", MIN_N_TIMEPOINTS)
@pytest.mark.parametrize("max_n_timepoints", MAX_N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", REGRESSION)
def test_make_example_multi_index_dataframe(
    n_cases, n_channels, min_n_timepoints, max_n_timepoints, n_classes, regression
):
    """Test generated multi-index data is in the correct format."""
    X, y = make_example_multi_index_dataframe(
        n_cases=n_cases,
        n_labels=n_classes,
        n_channels=n_channels,
        min_n_timepoints=min_n_timepoints,
        max_n_timepoints=max_n_timepoints,
        regression_target=regression,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    if min_n_timepoints == max_n_timepoints:
        assert X.shape == (n_cases * min_n_timepoints, n_channels)
    else:
        assert (
            X.shape[0] >= n_cases * min_n_timepoints
            and X.shape[0] <= n_cases * max_n_timepoints
        )
        assert X.shape[1] == n_channels
    assert y.shape == (n_cases,)
    assert get_type(X) == "pd-multiindex"
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes
