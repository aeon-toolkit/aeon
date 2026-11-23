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
    make_example_segment_series,
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


def test_make_example_segment_series_mean_shift():
    """Test generated mean shift segment series and verify actual shift."""
    n_timepoints = 3000
    mean_shift = 10.0
    noise_std = 0.5
    tolerance = 0.15

    X, cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        shift_type="mean",
        shift_magnitude=mean_shift,
        noise_std=noise_std,
        return_changepoints=True,
        random_state=42,
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints,)
    assert is_series(X)
    assert len(cps) == 2

    seg1 = X[: cps[0]]
    seg2 = X[cps[0] : cps[1]]
    seg3 = X[cps[1] :]

    assert abs(np.mean(seg1) - 0.0) < tolerance
    assert abs(np.mean(seg2) - mean_shift) < tolerance
    assert abs(np.mean(seg3) - 0.0) < tolerance
    assert abs(np.std(seg1) - noise_std) < tolerance
    assert abs(np.std(seg2) - noise_std) < tolerance
    assert abs(np.std(seg3) - noise_std) < tolerance


def test_make_example_segment_series_variance_shift():
    """Test generated variance shift segment series and verify actual shift."""
    n_timepoints = 3000
    baseline_std = 1.0
    shifted_std = 5.0
    tolerance = 0.25

    X, cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        shift_type="variance",
        shift_magnitude=shifted_std,
        noise_std=baseline_std,
        return_changepoints=True,
        random_state=7,
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_timepoints,)
    assert is_series(X)
    assert len(cps) == 2

    seg1 = X[: cps[0]]
    seg2 = X[cps[0] : cps[1]]
    seg3 = X[cps[1] :]

    assert abs(np.mean(seg1)) < tolerance
    assert abs(np.mean(seg2)) < tolerance
    assert abs(np.mean(seg3)) < tolerance
    assert abs(np.std(seg1) - baseline_std) < tolerance
    assert abs(np.std(seg2) - shifted_std) < tolerance
    assert abs(np.std(seg3) - baseline_std) < tolerance


@pytest.mark.parametrize("axis", [0, 1])
def test_make_example_segment_series_multivariate(axis):
    """Test generated multivariate segment series across axes."""
    n_timepoints = 60
    n_channels = 3

    X, cps = make_example_segment_series(
        n_timepoints=n_timepoints,
        n_channels=n_channels,
        axis=axis,
        return_changepoints=True,
        random_state=42,
    )

    expected_shape = (
        (n_timepoints, n_channels) if axis == 0 else (n_channels, n_timepoints)
    )
    assert X.shape == expected_shape
    assert is_series(X, include_2d=True)
    assert len(cps) == 2


def test_make_example_segment_series_invalid_inputs():
    """Test error handling for invalid arguments in make_example_segment_series."""
    with pytest.raises(ValueError, match="axis: 2 is not supported"):
        make_example_segment_series(axis=2)

    with pytest.raises(ValueError, match="shift_type 'unknown' is not supported"):
        make_example_segment_series(shift_type="unknown")

    with pytest.raises(ValueError, match="Invalid random_state"):
        make_example_segment_series(random_state="invalid")
