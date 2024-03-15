"""Test Truncator transformer."""

import numpy as np
import pytest

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
    load_unit_test,
)
from aeon.transformations.collection.truncate import TruncationTransformer


@pytest.mark.parametrize(
    "loader", [load_japanese_vowels, load_plaid, load_unit_test, load_basic_motions]
)
def test_truncation_transformer(loader):
    """Test truncation to the fixed series length on provided datasets."""
    X_train, y_train = loader(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    truncator = TruncationTransformer(5)
    Xt = truncator.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 5)


def test_truncation_equal_length():
    """Test the output dimensions after truncation.

    Truncation should not change shape of equal length unless passed parameter.
    """
    X = np.random.rand(10, 2, 20)
    truncator = TruncationTransformer()
    X_padded = truncator.fit_transform(X)
    assert X_padded.shape == X.shape


def test_truncation_parameterised_transformer():
    """Test padding to user determined length.

    Padding should change shape of equal length if passed longer value.
    """
    # load data
    X = np.random.rand(10, 2, 20)
    truncator = TruncationTransformer(truncated_length=10)
    X_padded = truncator.fit_transform(X)
    #  Series now of length 10
    assert X_padded.shape == (X.shape[0], X.shape[1], 10)


def test_truncation_fill_unequal_length():
    """Tes TruncationTransformer handles unequal length data correctly."""
    X = []
    for i in range(10):
        X.append(np.random.random((10, 15 + i)))
    truncator = TruncationTransformer(truncated_length=10)
    X_trunc = truncator.fit_transform(X)
    assert isinstance(X_trunc, np.ndarray)
    assert X_trunc.shape == (len(X), X[0].shape[0], 10)
