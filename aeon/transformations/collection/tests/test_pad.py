"""Test the Padder transformer."""

import numpy as np
import pytest

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
    load_unit_test,
)
from aeon.transformations.collection import Padder


@pytest.mark.parametrize(
    "loader", [load_japanese_vowels, load_plaid, load_unit_test, load_basic_motions]
)
def test_padding(loader):
    """Test padding to on provided datasets."""
    X_train, y_train = loader(split="train")
    n_cases = len(X_train)
    n_channels = len(X_train[0])
    pad = Padder(pad_length=2000)
    Xt = pad.fit_transform(X_train)
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (n_cases, n_channels, 2000)


def test_padding_equal_length():
    """Test the output dimensions after padding.

    Padding should not change shape of equal length unless passed parameter.
    """
    X = np.random.rand(10, 2, 20)
    padding_transformer = Padder()
    X_padded = padding_transformer.fit_transform(X)
    assert X_padded.shape == X.shape


def test_padding_parameterised_transformer():
    """Test padding to user determined length.

    Padding should change shape of equal length if passed longer value.
    """
    # load data
    X = np.random.rand(10, 2, 20)
    padding_transformer = Padder(pad_length=120)
    X_padded = padding_transformer.fit_transform(X)
    #  Series now of length 120
    assert X_padded.shape == (X.shape[0], X.shape[1], 120)


def test_padding_fixed_value():
    """Test full fill padding.

    Padding should change set a value if passed.
    """
    X = np.random.rand(10, 2, 20)
    padding_transformer = Padder(pad_length=120, fill_value=42)
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
    padding_transformer = Padder(pad_length=120, fill_value=42)
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
    padding_transformer = Padder(pad_length=10, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)
    assert isinstance(X_padded, np.ndarray)
    assert X_padded.shape == (len(X), X[0].shape[0], 24)


@pytest.mark.parametrize(
    "fill_value",
    ["mean", "median", "max", "min", np.random.random(size=(10, 2))],
)
def test_fill_value_with_string_params(fill_value):
    """Test if the fill_value argument returns  the correct results."""
    X = np.random.rand(10, 2, 20)
    padding_transformer = Padder(pad_length=120, fill_value=fill_value)
    X_padded = padding_transformer.fit_transform(X)

    assert X_padded.shape == (X.shape[0], X.shape[1], 120)


def test_padder_incorrect_paras():
    """Test Padder with incorrect parameters."""
    X = np.random.rand(2, 2, 20)
    padding_transformer = Padder(pad_length=22, fill_value="FOOBAR")
    with pytest.raises(ValueError, match="Supported modes are mean, median, min, max"):
        padding_transformer.fit_transform(X)
    padding_transformer = Padder(pad_length=22, fill_value=np.array([1, 2, 3, 4]))
    with pytest.raises(ValueError, match="The length of fill_value must match"):
        padding_transformer.fit_transform(X)
    padding_transformer = Padder(pad_length=22, fill_value=np.array([1, 2]))
    with pytest.raises(ValueError, match="The fill_value argument must be"):
        padding_transformer.fit_transform(X)
    X2 = np.random.rand(2, 2, 50)
    padding_transformer = Padder(pad_length=22)
    with pytest.raises(ValueError, match="max_length of series"):
        padding_transformer.fit(X)
        padding_transformer.transform(X2)
