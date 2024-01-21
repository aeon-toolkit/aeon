"""Test the Padder transformer."""

import numpy as np
import pytest

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
    load_unit_test,
)
from aeon.transformations.collection.pad import PaddingTransformer


@pytest.mark.parametrize(
    "loader", [load_japanese_vowels, load_plaid, load_unit_test, load_basic_motions]
)
def test_padding(loader):
    """Test padding to on provided datasets."""
    X_train, y_train = loader(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    pad = PaddingTransformer(pad_length=2000)
    Xt = pad.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 2000)


def test_padding_equal_length():
    """Test the output dimensions after padding.

    Padding should not change shape of equal length unless passed parameter.
    """
    X = np.random.rand(10, 2, 20)
    padding_transformer = PaddingTransformer()
    X_padded = padding_transformer.fit_transform(X)
    assert X_padded.shape == X.shape


def test_padding_parameterised_transformer():
    """Test padding to user determined length.

    Padding should change shape of equal length if passed longer value.
    """
    # load data
    X = np.random.rand(10, 2, 20)
    padding_transformer = PaddingTransformer(pad_length=120)
    X_padded = padding_transformer.fit_transform(X)
    #  Series now of length 120
    assert X_padded.shape == (X.shape[0], X.shape[1], 120)


def test_padding_fixed_value():
    """Test full fill padding.

    Padding should change set a value if passed.
    """
    X = np.random.rand(10, 2, 20)
    padding_transformer = PaddingTransformer(pad_length=120, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)

    assert X_padded.shape == (X.shape[0], X.shape[1], 120)
    assert X_padded[0][0][100] == 42


def test_padding_fill_unequal_length():
    """Test padding unequal length longer than longest.

    Padding should create a 3D numpy array and padd to given value.
    """
    X = []
    for i in range(10):
        X.append(np.random.random((10, 15 + i)))
    padding_transformer = PaddingTransformer(pad_length=120, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)
    assert isinstance(X_padded, np.ndarray)
    assert X_padded.shape == (len(X), X[0].shape[0], 120)


def test_padding_fill_too_short_pad_value():
    """Test padding unequal length shorter than longest.

    If passed a value shorter than longest, pad should bad to longest.
    """
    X = []
    for i in range(10):
        X.append(np.random.random((10, 15 + i)))
    padding_transformer = PaddingTransformer(pad_length=10, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)
    assert isinstance(X_padded, np.ndarray)
    assert X_padded.shape == (len(X), X[0].shape[0], 24)
