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
from aeon.utils.validation.series import is_series

N_CHANNELS = [1, 3]
N_TIMEPOINTS = [6, 10]


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_example_1d_numpy(n_timepoints):
    """Test generated 1d numpy data is in the correct format."""
    X = make_example_1d_numpy(n_timepoints=n_timepoints)

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints,)
    assert is_series(X)

    X, y = make_example_1d_numpy(n_timepoints=n_timepoints, return_y="anomaly")

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints,)
    assert is_series(X)
    assert isinstance(y, np.ndarray)
    assert y.shape == (n_timepoints,)
    assert np.all(np.isin(y, [0, 1]))

    with pytest.raises(ValueError, match="value for return_y is not supported"):
        make_example_1d_numpy(n_timepoints=n_timepoints, return_y="invalid")


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
def test_make_example_2d_numpy_series(n_timepoints, n_channels):
    """Test generated 2d numpy data is in the correct format."""
    X = make_example_2d_numpy_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=0
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints, n_channels)
    assert is_series(X, include_2d=True)

    X = make_example_2d_numpy_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=1
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_channels, n_timepoints)
    assert is_series(X, include_2d=True)

    (
        X,
        y,
    ) = make_example_2d_numpy_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=1, return_y="anomaly"
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_channels, n_timepoints)
    assert is_series(X, include_2d=True)
    assert isinstance(y, np.ndarray)
    assert y.shape == (n_timepoints,)
    assert np.all(np.isin(y, [0, 1]))

    with pytest.raises(ValueError, match="value for return_y is not supported"):
        make_example_2d_numpy_series(
            n_timepoints=n_timepoints, n_channels=n_channels, axis=1, return_y="invalid"
        )


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_example_pandas_series(n_timepoints):
    """Test generated series data is in the correct format."""
    X = make_example_pandas_series(n_timepoints=n_timepoints)

    assert isinstance(X, pd.Series)
    assert X.shape == (n_timepoints,)
    assert is_series(X)

    X, y = make_example_pandas_series(n_timepoints=n_timepoints, return_y="anomaly")

    assert isinstance(X, pd.Series)
    assert X.shape == (n_timepoints,)
    assert is_series(X)
    assert isinstance(y, np.ndarray)
    assert y.shape == (n_timepoints,)
    assert np.all(np.isin(y, [0, 1]))

    with pytest.raises(ValueError, match="value for return_y is not supported"):
        make_example_pandas_series(n_timepoints=n_timepoints, return_y="invalid")


@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
def test_make_example_dataframe_series(n_timepoints, n_channels):
    """Test generated dataframe data is in the correct format."""
    X = make_example_dataframe_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=0
    )

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n_timepoints, n_channels)
    assert is_series(X, include_2d=True)

    X = make_example_dataframe_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=1
    )

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n_channels, n_timepoints)
    assert is_series(X, include_2d=True)

    (
        X,
        y,
    ) = make_example_dataframe_series(
        n_timepoints=n_timepoints, n_channels=n_channels, axis=1, return_y="anomaly"
    )

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n_channels, n_timepoints)
    assert is_series(X, include_2d=True)
    assert isinstance(y, np.ndarray)
    assert y.shape == (n_timepoints,)
    assert np.all(np.isin(y, [0, 1]))

    with pytest.raises(ValueError, match="value for return_y is not supported"):
        make_example_dataframe_series(
            n_timepoints=n_timepoints, n_channels=n_channels, axis=1, return_y="invalid"
        )
