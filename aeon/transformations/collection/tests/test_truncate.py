"""Test Truncator transformer."""

import numpy as np
import pytest

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
    load_unit_test,
)
from aeon.transformations.collection import Truncator


@pytest.mark.parametrize(
    "loader", [load_japanese_vowels, load_plaid, load_unit_test, load_basic_motions]
)
def test_truncation_transformer(loader):
    """Test truncation to the fixed series length on provided datasets."""
    X_train, y_train = loader(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    truncator = Truncator(5)
    Xt = truncator.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 5)


def test_truncation_equal_length():
    """Test the output dimensions after truncation.

    Truncation should not change shape of equal length unless passed parameter.
    """
    X = np.random.rand(10, 2, 20)
    truncator = Truncator()
    Xt = truncator.fit_transform(X)
    assert Xt.shape == X.shape


def test_truncation_parameterised_transformer():
    """Test padding to user determined length.

    Padding should change shape of equal length if passed longer value.
    """
    # load data
    X = np.random.rand(10, 2, 20)
    truncator = Truncator(truncated_length=10)
    Xt = truncator.fit_transform(X)
    #  Series now of length 10
    assert Xt.shape == (X.shape[0], X.shape[1], 10)


def test_truncation_fill_unequal_length():
    """Tes Truncator handles unequal length data correctly."""
    X = []
    for i in range(10):
        X.append(np.random.random((10, 15 + i)))
    truncator = Truncator(truncated_length=10)
    Xt = truncator.fit_transform(X)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (len(X), X[0].shape[0], 10)


def test_incorrect_arguments():
    """Test Truncator with incorrect constructor arguments."""
    X = np.random.rand(10, 1, 20)
    truncator = Truncator(truncated_length=30)
    truncator.fit(X)
    assert truncator.truncated_length_ == 20
    X2 = np.random.rand(10, 1, 10)
    with pytest.raises(ValueError, match="min_length of series"):
        truncator.transform(X2)
