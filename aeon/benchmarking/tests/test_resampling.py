"""Tests for dataset resampling functions."""

import numpy as np
import pandas as pd
import pytest

from aeon.benchmarking.resampling import (
    resample_data,
    resample_data_indices,
    stratified_resample_data,
    stratified_resample_data_indices,
)
from aeon.datasets import load_unit_test
from aeon.testing.testing_data import (
    EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION,
    EQUAL_LENGTH_UNIVARIATE_REGRESSION,
    UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION,
    UNEQUAL_LENGTH_UNIVARIATE_REGRESSION,
)


@pytest.mark.parametrize(
    "data",
    [
        EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"],
        EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION["numpy3D"],
    ],
)
def test_resample_data(data):
    """Test resampling returns valid data."""
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    train_size = X_train.shape
    test_size = X_test.shape

    X_train, y_train, X_test, y_test = resample_data(X_train, y_train, X_test, y_test)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.shape == train_size
    assert X_test.shape == test_size


@pytest.mark.parametrize(
    "data",
    [
        UNEQUAL_LENGTH_UNIVARIATE_REGRESSION["np-list"],
        UNEQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION["np-list"],
    ],
)
def test_resample_data_unequal(data):
    """Test resampling returns valid data with unequal length input."""
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    train_size = len(X_train)
    test_size = len(X_test)

    X_train, y_train, X_test, y_test = resample_data(X_train, y_train, X_test, y_test)

    assert isinstance(X_train, list)
    assert isinstance(X_train[0], np.ndarray)
    assert isinstance(X_test, list)
    assert isinstance(X_test[0], np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert len(X_train) == train_size
    assert len(X_test) == test_size


def test_resample_data_invalid():
    """Test resampling raises an error with invalid input."""
    X = pd.DataFrame(np.random.random((10, 10)))
    y = pd.Series(np.zeros(10))

    with pytest.raises(ValueError, match="X_train must be a"):
        resample_data(X, y, X, y)


def test_resample_data_indices():
    """Test resampling returns valid indices."""
    X_train, y_train = load_unit_test(split="TRAIN")
    X_test, y_test = load_unit_test(split="TEST")

    new_X_train, _, new_X_test, _ = resample_data(
        X_train, y_train, X_test, y_test, random_state=0
    )
    train_indices, test_indices = resample_data_indices(y_train, y_test, random_state=0)
    X = np.concatenate((X_train, X_test), axis=0)

    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert len(train_indices) == len(new_X_train)
    assert len(test_indices) == len(new_X_test)
    assert len(np.unique(np.concatenate((train_indices, test_indices), axis=0))) == len(
        X
    )
    assert (new_X_train[0] == X[train_indices[0]]).all()
    assert (new_X_test[0] == X[test_indices[0]]).all()

    concat = np.concatenate((train_indices, test_indices), axis=0)
    assert len(np.unique(concat)) == len(concat)

    # expected indicies after resampling
    np.testing.assert_array_equal(
        concat,
        [
            30,
            36,
            27,
            4,
            10,
            25,
            28,
            11,
            37,
            31,
            29,
            20,
            39,
            2,
            41,
            18,
            15,
            22,
            16,
            38,
            8,
            13,
            5,
            17,
            32,
            14,
            35,
            7,
            34,
            1,
            26,
            12,
            33,
            24,
            6,
            23,
            21,
            19,
            9,
            40,
            3,
            0,
        ],
    )


@pytest.mark.parametrize(
    "data",
    [
        EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"],
        EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION["numpy3D"],
    ],
)
def test_stratified_resample_data(data):
    """Test stratified resampling returns valid data and class distribution."""
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    train_size = X_train.shape
    test_size = X_test.shape
    _, counts_train = np.unique(y_train, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)

    X_train, y_train, X_test, y_test = stratified_resample_data(
        X_train, y_train, X_test, y_test
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.shape == train_size
    assert X_test.shape == test_size

    _, counts_train_new = np.unique(y_train, return_counts=True)
    _, counts_test_new = np.unique(y_test, return_counts=True)

    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)


@pytest.mark.parametrize(
    "data",
    [
        EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["np-list"],
        EQUAL_LENGTH_MULTIVARIATE_CLASSIFICATION["np-list"],
    ],
)
def test_stratified_resample_data_unequal(data):
    """Test stratified resampling returns valid data with unequal length input."""
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]

    train_size = len(X_train)
    test_size = len(X_test)
    _, counts_train = np.unique(y_train, return_counts=True)
    _, counts_test = np.unique(y_test, return_counts=True)

    X_train, y_train, X_test, y_test = stratified_resample_data(
        X_train, y_train, X_test, y_test
    )

    assert isinstance(X_train, list)
    assert isinstance(X_train[0], np.ndarray)
    assert isinstance(X_test, list)
    assert isinstance(X_test[0], np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert len(X_train) == train_size
    assert len(X_test) == test_size

    _, counts_train_new = np.unique(y_train, return_counts=True)
    _, counts_test_new = np.unique(y_test, return_counts=True)

    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)


def test_stratified_resample_data_regression():
    """Test stratified resampling returns valid data for regression."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"]["train"]
    X_test, y_test = EQUAL_LENGTH_UNIVARIATE_REGRESSION["numpy3D"]["test"]

    with pytest.raises(ValueError, match="not valid for classification"):
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test
        )


def test_stratified_resample_data_invalid():
    """Test stratified resampling raises an error with invalid input."""
    X = pd.DataFrame(np.random.random((10, 10)))
    y = pd.Series(np.zeros(10))

    with pytest.raises(ValueError, match="X_train must be a"):
        stratified_resample_data(X, y, X, y)


def test_stratified_resample_data_indices():
    """Test stratified resampling returns valid indices."""
    X_train, y_train = load_unit_test(split="TRAIN")
    X_test, y_test = load_unit_test(split="TEST")

    new_X_train, _, new_X_test, _ = stratified_resample_data(
        X_train, y_train, X_test, y_test, random_state=0
    )
    train_indices, test_indices = stratified_resample_data_indices(
        y_train, y_test, random_state=0
    )
    X = np.concatenate((X_train, X_test), axis=0)

    assert isinstance(train_indices, np.ndarray)
    assert isinstance(test_indices, np.ndarray)
    assert len(train_indices) == len(new_X_train)
    assert len(test_indices) == len(new_X_test)
    assert len(np.unique(np.concatenate((train_indices, test_indices), axis=0))) == len(
        X
    )
    assert (new_X_train[0] == X[train_indices[0]]).all()
    assert (new_X_test[0] == X[test_indices[0]]).all()

    concat = np.concatenate((train_indices, test_indices), axis=0)
    assert len(np.unique(concat)) == len(concat)

    # expected indicies after resampling
    np.testing.assert_array_equal(
        concat,
        [
            30,
            20,
            24,
            23,
            1,
            31,
            21,
            29,
            8,
            6,
            16,
            36,
            38,
            32,
            17,
            11,
            37,
            33,
            12,
            34,
            26,
            4,
            2,
            5,
            27,
            9,
            7,
            28,
            3,
            0,
            25,
            22,
            40,
            41,
            10,
            13,
            14,
            19,
            18,
            35,
            15,
            39,
        ],
    )
