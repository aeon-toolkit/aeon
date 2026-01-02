"""Test the Padder transformer."""

import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_pickup_gesture_wiimoteZ,
    load_unit_test,
)
from aeon.testing.data_generation import make_example_3d_numpy_list
from aeon.testing.utils.deep_equals import deep_equals
from aeon.transformations.collection.unequal_length import Padder
from aeon.transformations.collection.unequal_length._commons import _get_min_length


@pytest.mark.parametrize(
    "loader",
    [
        load_japanese_vowels,
        load_pickup_gesture_wiimoteZ,
        load_unit_test,
        load_basic_motions,
    ],
)
def test_padding(loader):
    """Test padding to on provided datasets."""
    X_train, y_train = loader(split="train")
    pad = Padder(padded_length=2000)
    Xt = pad.fit_transform(X_train)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (len(X_train), len(X_train[0]), 2000)

    for i in range(5):
        assert_array_equal(X_train[i], Xt[i, :, : len(X_train[i][0])])


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
    padding_transformer = Padder(padded_length=120)
    X_padded = padding_transformer.fit_transform(X)
    #  Series now of length 120
    assert X_padded.shape == (X.shape[0], X.shape[1], 120)


def test_padding_fixed_value():
    """Test full fill padding.

    Padding should change set a value if passed.
    """
    X = np.random.rand(10, 2, 20)
    padding_transformer = Padder(padded_length=120, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)

    assert X_padded.shape == (X.shape[0], X.shape[1], 120)
    assert X_padded[0][0][100] == 42


def test_padding_fill_unequal_length():
    """Test padding unequal length longer than longest.

    Padding should create a 3D numpy array and pad to given value.
    """
    X = []
    for i in range(10):
        X.append(np.random.random((10, 15 + i)))
    padding_transformer = Padder(padded_length=120, fill_value=42)
    X_padded = padding_transformer.fit_transform(X)
    assert isinstance(X_padded, np.ndarray)
    assert X_padded.shape == (len(X), X[0].shape[0], 120)


def test_padding_min():
    """Test Truncator handles unequal length data correctly."""
    X, _ = make_example_3d_numpy_list()
    X2, _ = make_example_3d_numpy_list(min_n_timepoints=6, max_n_timepoints=16)
    min_length = _get_min_length(X)

    padder = Padder(padded_length="min", error_on_long=False)
    padder.fit(X)
    Xt = padder.transform(X2)

    assert isinstance(Xt, list)
    assert len(Xt) == len(X2)
    assert len(Xt[0].shape) == 2 and Xt[0].shape[0] == 1
    assert all(
        (
            Xt[i].shape[1] == min_length
            if X2[i].shape[1] < min_length
            else Xt[i].shape[1] == X2[i].shape[1]
        )
        for i in range(len(Xt))
    )

    X3, _ = make_example_3d_numpy_list(min_n_timepoints=13, max_n_timepoints=17)
    Xt2 = padder.transform(X3)
    assert deep_equals(Xt2, X3)


def test_padding_fill_too_short_pad_value():
    """Test padding unequal length shorter than longest."""
    X = np.random.rand(2, 2, 50)

    padding_transformer = Padder(padded_length=22)
    with pytest.raises(ValueError, match="max length of series"):
        padding_transformer.fit(X)
        padding_transformer.transform(X)


@pytest.mark.parametrize(
    "fill_value",
    ["mean", "median", "max", "min", "last"],
)
def test_fill_value_with_string_params(fill_value):
    """Test if the fill_value string arguments run without error."""
    X = np.random.rand(10, 2, 20)
    padding_transformer = Padder(padded_length=120, fill_value=fill_value)
    X_padded = padding_transformer.fit_transform(X)
    assert X_padded.shape == (X.shape[0], X.shape[1], 120)


def test_padder_incorrect_paras():
    """Test Padder with incorrect parameters."""
    X = np.random.rand(2, 2, 20)

    padding_transformer = Padder(padded_length=22, fill_value="FOOBAR")
    with pytest.raises(ValueError, match="Supported str values for fill_value are"):
        padding_transformer.fit_transform(X)
