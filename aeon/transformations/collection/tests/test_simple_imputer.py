"""Test for SimpleImputer."""

__maintainer__ = []

import numpy as np
import pytest

from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.transformations.collection import SimpleImputer


def test_3d_numpy():
    """Test SimpleImputer with 3D numpy array."""
    X, _ = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=50, random_state=42
    )
    X[2, 1, 10] = np.nan
    X[5, 0, 20] = np.nan

    imputer = SimpleImputer(strategy="mean")
    Xt = imputer.fit_transform(X)

    assert not np.isnan(Xt).any()
    assert Xt.shape == X.shape
    assert np.allclose(Xt[2, 1, 10], np.nanmean(X[2, 1, :]))


def test_2d_list():
    """Test SimpleImputer with 2D list."""
    X, _ = make_example_3d_numpy_list(
        n_cases=5,
        n_channels=2,
        min_n_timepoints=50,
        max_n_timepoints=70,
        random_state=42,
    )
    X[2][1, 10] = np.nan
    X[4][0, 20] = np.nan

    imputer = SimpleImputer(strategy="mean")
    Xt = imputer.fit_transform(X)

    assert all(not np.isnan(x).any() for x in Xt)  # no NaNs in any of the arrays
    assert Xt[2][1, 10] == np.nanmean(X[2][1, :])
    assert Xt[4][0, 20] == np.nanmean(X[4][0, :])


def test_median():
    """Test SimpleImputer with median strategy."""
    X, _ = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=50, random_state=42
    )
    X[2, 1, 10] = np.nan
    X[5, 0, 20] = np.nan

    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X)

    assert not np.isnan(Xt).any()
    assert Xt.shape == X.shape
    assert np.allclose(Xt[2, 1, 10], np.nanmedian(X[2, 1, :]))
    assert np.allclose(Xt[5, 0, 20], np.nanmedian(X[5, 0, :]))


def test_most_frequent():
    """Test SimpleImputer with most frequent strategy."""
    from scipy.stats import mode

    X, _ = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=50, random_state=42
    )
    X[2, 1, 10] = np.nan
    X[5, 0, 20] = np.nan

    imputer = SimpleImputer(strategy="most frequent")
    Xt = imputer.fit_transform(X)

    assert not np.isnan(Xt).any()
    assert Xt.shape == X.shape
    assert np.allclose(Xt[2, 1, 10], mode(X[2, 1, :], nan_policy="omit").mode)
    assert np.allclose(Xt[5, 0, 20], mode(X[5, 0, :], nan_policy="omit").mode)


def test_constant():
    """Test SimpleImputer with constant strategy."""
    X, _ = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=50, random_state=42
    )
    X[2, 1, 10] = np.nan
    X[5, 0, 20] = np.nan

    imputer = SimpleImputer(strategy="constant", fill_value=-1)
    Xt = imputer.fit_transform(X)

    assert not np.isnan(Xt).any()
    assert Xt.shape == X.shape
    assert np.allclose(Xt[2, 1, 10], -1)
    assert np.allclose(Xt[5, 0, 20], -1)


def test_valid_parameters():
    """Test SimpleImputer with valid parameters."""
    X, _ = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=50, random_state=42
    )

    imputer = SimpleImputer(strategy="constant")  # no fill_value

    with pytest.raises(ValueError):
        imputer.fit_transform(X)

    imputer = SimpleImputer(strategy="mode")  # invalid strategy

    with pytest.raises(ValueError):
        imputer.fit_transform(X)


def test_callable():
    """Test SimpleImputer with callable strategy."""
    X, _ = make_example_3d_numpy(
        n_cases=10, n_channels=2, n_timepoints=50, random_state=42
    )
    X[2, 1, 10] = np.nan
    X[5, 0, 20] = np.nan

    def dummy_strategy(x):
        return 0

    imputer = SimpleImputer(strategy=dummy_strategy)
    Xt = imputer.fit_transform(X)

    assert not np.isnan(Xt).any()
    assert Xt.shape == X.shape
    assert np.allclose(Xt[2, 1, 10], 0)
    assert np.allclose(Xt[5, 0, 20], 0)
