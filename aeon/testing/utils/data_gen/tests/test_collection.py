"""Tests for datagen functions."""

__maintainer__ = []

import numpy as np
import pandas as pd
import pytest
from numpy import array_equal

from aeon.testing.utils.data_gen import (
    _make_classification_y,
    _make_nested_from_array,
    _make_regression_y,
    make_example_2d_numpy,
    make_example_2d_unequal_length,
    make_example_3d_numpy,
    make_example_nested_dataframe,
    make_example_unequal_length,
    piecewise_poisson,
)

N_CASES = [10]
N_CHANNELS = [1, 3]
N_TIMEPOINTS = [3, 5]
N_CLASSES = [2, 5]


def _check_X_y_pandas(X, y, n_cases, n_columns, n_timepoints):
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0] == n_cases
    assert X.shape[1] == n_columns
    assert X.iloc[0, 0].shape == (n_timepoints,)


def _check_X_y_numpy(X, y, n_cases, n_columns, n_timepoints):
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_cases, n_columns, n_timepoints)
    assert y.shape == (n_cases,)


def _check_X_y(X, y, n_cases, n_columns, n_timepoints, check_numpy=False):
    if check_numpy:
        _check_X_y_numpy(X, y, n_cases, n_columns, n_timepoints)
    else:
        _check_X_y_pandas(X, y, n_cases, n_columns, n_timepoints)


@pytest.mark.parametrize(
    "lambdas, lengths, random_state, output",
    [
        ([1, 2, 3], [2, 4, 8], 42, [1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 2, 4, 2, 1]),
        ([1, 3, 6], [2, 4, 8], 42, [1, 2, 1, 3, 3, 2, 5, 5, 6, 4, 4, 9, 3, 5]),
    ],
)
def test_piecewise_poisson(lambdas, lengths, random_state, output):
    """Test piecewise_poisson fuction returns the expected Poisson distributed array."""
    assert array_equal(piecewise_poisson(lambdas, lengths, random_state), output)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", [True, False])
def test_make_example_3d_numpy(
    n_cases, n_channels, n_timepoints, n_classes, regression
):
    """Test data of right format."""
    X, y = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_labels=n_classes,
        regression_target=regression,
    )
    _check_X_y(X, y, n_cases, n_channels, n_timepoints, check_numpy=True)
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", [True, False])
def test_make_example_2d_numpy(n_cases, n_timepoints, n_classes, regression):
    """Test data of right format."""
    X, y = make_example_2d_numpy(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
        n_labels=n_classes,
        regression_target=regression,
    )
    assert X.shape == (n_cases, n_timepoints)
    assert len(y) == len(X)
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", [True, False])
def test_make_unequal_length_data(
    n_cases, n_channels, n_timepoints, n_classes, regression
):
    """Test data of right format."""
    X, y = make_example_unequal_length(
        n_cases=n_cases,
        n_channels=n_channels,
        n_labels=n_classes,
        min_n_timepoints=n_timepoints - 1,
        max_n_timepoints=n_timepoints + 1,
        regression_target=regression,
    )
    assert isinstance(X, list)
    assert len(X) == len(y) == n_cases
    assert X[0].shape[0] == n_channels
    assert abs(X[0].shape[1] - n_timepoints) <= 1
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", [True, False])
def test_make_2d_unequal_length_data(n_cases, n_timepoints, n_classes, regression):
    """Test data of right format."""
    X, y = make_example_2d_unequal_length(
        n_cases=n_cases,
        min_n_timepoints=n_timepoints - 1,
        max_n_timepoints=n_timepoints + 1,
        n_labels=n_classes,
        regression_target=regression,
    )
    assert isinstance(X, list)
    assert len(X) == len(y) == n_cases
    assert abs(X[0].shape[0] - n_timepoints) <= 1
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("regression", [True, False])
def test_make_example_nested_dataframe(
    n_cases, n_channels, n_timepoints, n_classes, regression
):
    """Test examples nested dataframes creation."""
    X, y = make_example_nested_dataframe(
        n_cases=n_cases,
        n_labels=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        regression_target=regression,
    )

    # check dimensions of generated data
    _check_X_y(X, y, n_cases, n_channels, n_timepoints, check_numpy=False)
    if regression:
        assert y.dtype == np.float32
    else:
        assert len(np.unique(y)) == n_classes


def test_uncovered():
    """Test data gen cases we dont need to do for all combos."""
    X, y = make_example_2d_numpy(n_cases=4, n_labels=5)
    assert np.unique(y).size == 4
    y = _make_regression_y(return_numpy=True)
    assert isinstance(y, np.ndarray)
    y = _make_regression_y(return_numpy=False)
    assert isinstance(y, pd.Series)
    with pytest.raises(ValueError, match="n_cases must be bigger than n_classes"):
        y = _make_classification_y(n_cases=4, n_classes=5)
    x = _make_nested_from_array(make_example_3d_numpy(n_channels=2), 2)
    assert isinstance(x, pd.DataFrame)
