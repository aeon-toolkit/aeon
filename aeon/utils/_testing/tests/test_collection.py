# -*- coding: utf-8 -*-
"""Tests for data and scenario generators in _testing.panel module."""

__author__ = ["mloning", "fkiraly", "TonyBagnall"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from aeon.utils._testing.collection import (
    make_2d_test_data,
    make_3d_test_data,
    make_clustering_data,
    make_nested_dataframe_data,
    make_unequal_length_test_data,
)

N_INSTANCES = [10, 15]
N_CHANNELS = [3, 5]
N_TIMEPOINTS = [3, 5]
N_CLASSES = [2, 5]


def _check_X_y_pandas(X, y, n_instances, n_columns, n_timepoints):
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0] == n_instances
    assert X.shape[1] == n_columns
    assert X.iloc[0, 0].shape == (n_timepoints,)


def _check_X_y_numpy(X, y, n_instances, n_columns, n_timepoints):
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (n_instances, n_columns, n_timepoints)
    assert y.shape == (n_instances,)


def _check_X_y(X, y, n_instances, n_columns, n_timepoints, check_numpy=False):
    if check_numpy:
        _check_X_y_numpy(X, y, n_instances, n_columns, n_timepoints)
    else:
        _check_X_y_pandas(X, y, n_instances, n_columns, n_timepoints)


@pytest.mark.parametrize("n_cases", N_INSTANCES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_3d_test_data(n_cases, n_channels, n_timepoints):
    """Test data of right format."""
    X, y = make_3d_test_data(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
    )
    _check_X_y(X, y, n_cases, n_channels, n_timepoints, check_numpy=True)


@pytest.mark.parametrize("n_cases", N_INSTANCES)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_2d_test_data(n_cases, n_timepoints):
    """Test data of right format."""
    X, y = make_2d_test_data(
        n_cases=n_cases,
        n_timepoints=n_timepoints,
    )
    assert X.shape == (n_cases, n_timepoints)
    assert len(y) == len(X)


@pytest.mark.parametrize("n_cases", N_INSTANCES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
@pytest.mark.parametrize("n_classes", N_CLASSES)
@pytest.mark.parametrize("classification", [True, False])
def test_make_dataframe_data(
    n_cases, n_channels, n_timepoints, n_classes, classification
):
    X, y = make_nested_dataframe_data(
        n_cases=n_cases,
        n_classes=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        classification=classification,
    )

    # check dimensions of generated data
    _check_X_y(X, y, n_cases, n_channels, n_timepoints, check_numpy=False)
    if classification:
        assert len(np.unique(y)) == n_classes
    else:
        assert type(y) == pd.Series


@pytest.mark.parametrize("n_cases", N_INSTANCES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_clustering_data(
    n_cases,
    n_channels,
    n_timepoints,
):
    X = make_clustering_data(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
    )
    assert X.shape == (n_cases, n_channels, n_timepoints)


@pytest.mark.parametrize("n_cases", N_INSTANCES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_make_unequal_length_test_data(n_cases, n_channels, n_timepoints):
    """Test data of right format."""
    X, y = make_unequal_length_test_data(
        n_cases=n_cases,
        n_channels=n_channels,
        min_series_length=n_timepoints - 1,
        max_series_length=n_timepoints + 1,
    )
    assert isinstance(X, list)
    assert len(X) == len(y) == n_cases
    assert X[0].shape[0] == n_channels
    assert abs(X[0].shape[1] - n_timepoints) <= 1
