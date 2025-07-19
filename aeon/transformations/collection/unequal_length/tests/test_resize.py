"""Test Resizer transformer."""

import numpy as np
import pytest

from aeon.datasets import (
    load_basic_motions,
    load_japanese_vowels,
    load_pickup_gesture_wiimoteZ,
    load_unit_test,
)
from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.transformations.collection.unequal_length import Resizer
from aeon.transformations.collection.unequal_length._commons import (
    _get_max_length,
    _get_min_length,
)


@pytest.mark.parametrize(
    "loader",
    [
        load_japanese_vowels,
        load_pickup_gesture_wiimoteZ,
        load_unit_test,
        load_basic_motions,
    ],
)
def test_resizer_transformer(loader):
    """Test resizing to the fixed series length on provided datasets."""
    X, _ = loader(split="train")

    resizer = Resizer(resized_length=10)
    Xt = resizer.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (len(X), len(X[0]), 10)


def test_resizer_equal_length():
    """Test the output dimensions after resizing.

    Resizing should not change shape of equal length unless passed parameter.
    """
    X, _ = make_example_3d_numpy()

    resizer = Resizer()
    Xt = resizer.fit_transform(X)

    assert Xt.shape == X.shape


def test_resizer_fill_unequal_length():
    """Test Resizer handles unequal length data correctly."""
    X, _ = make_example_3d_numpy_list()

    resizer = Resizer()
    Xt = resizer.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (10, 1, _get_max_length(X))


def test_resizer_min():
    """Test Resizer handles unequal length data correctly."""
    X, _ = make_example_3d_numpy_list()
    X2, _ = make_example_3d_numpy_list(min_n_timepoints=6, max_n_timepoints=16)
    min_length = _get_min_length(X)

    resizer = Resizer(resized_length="min")
    resizer.fit(X)
    Xt = resizer.transform(X2)

    assert isinstance(Xt, np.ndarray)
    assert len(Xt) == len(X2)
    assert len(Xt[0].shape) == 2 and Xt[0].shape[0] == 1
    assert all((Xt[i].shape[1] == min_length) for i in range(len(Xt)))


def test_incorrect_arguments():
    """Test Resizer with incorrect constructor arguments."""
    X, _ = make_example_3d_numpy()

    resizer = Resizer(resized_length="invalid")
    with pytest.raises(ValueError, match="resized_length must be"):
        resizer.fit_transform(X)
