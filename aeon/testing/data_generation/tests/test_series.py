"""Tests for series data generation functions."""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
import pandas as pd
import pytest

from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_dataframe_series,
    make_example_pandas_series,
)
from aeon.utils.validation import is_single_series

N_CHANNELS = [1, 3]
N_TIMEPOINTS = [6, 10]


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_example_1d_numpy(n_timepoints):
    """Test generated 1d numpy data is in the correct format."""
    X = make_example_1d_numpy(n_timepoints=n_timepoints)

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints,)
    assert is_single_series(X)


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
def test_make_example_2d_numpy_series(n_timepoints, n_channels):
    """Test generated 2d numpy data is in the correct format."""
    X = make_example_2d_numpy_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=0
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints, n_channels)
    assert is_single_series(X)

    X = make_example_2d_numpy_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=1
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_channels, n_timepoints)
    assert is_single_series(X)


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_example_pandas_series(n_timepoints):
    """Test generated series data is in the correct format."""
    X = make_example_pandas_series(n_timepoints=n_timepoints)

    assert isinstance(X, pd.Series)
    assert X.shape == (n_timepoints,)
    assert is_single_series(X)


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
def test_make_example_dataframe_series(n_timepoints, n_channels):
    """Test generated dataframe data is in the correct format."""
    X = make_example_dataframe_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=0
    )

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n_timepoints, n_channels)
    assert is_single_series(X)

    X = make_example_dataframe_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=1
    )

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n_channels, n_timepoints)
    assert is_single_series(X)
