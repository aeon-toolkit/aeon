"""Test Truncator transformer."""

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
from aeon.testing.utils.deep_equals import deep_equals
from aeon.transformations.collection.unequal_length import Truncator
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
def test_truncation_transformer(loader):
    """Test truncation to the fixed series length on provided datasets."""
    X, _ = loader(split="train")

    truncator = Truncator(truncated_length=5)
    Xt = truncator.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (len(X), len(X[0]), 5)


def test_truncation_equal_length():
    """Test the output dimensions after truncation.

    Truncation should not change shape of equal length unless passed parameter.
    """
    X, _ = make_example_3d_numpy()

    truncator = Truncator()
    Xt = truncator.fit_transform(X)

    assert Xt.shape == X.shape


def test_truncation_fill_unequal_length():
    """Test Truncator handles unequal length data correctly."""
    X, _ = make_example_3d_numpy_list()

    truncator = Truncator()
    Xt = truncator.fit_transform(X)

    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (10, 1, _get_min_length(X))


def test_truncation_max():
    """Test Truncator handles unequal length data correctly."""
    X, _ = make_example_3d_numpy_list()
    X2, _ = make_example_3d_numpy_list(min_n_timepoints=6, max_n_timepoints=16)
    max_length = _get_max_length(X)

    truncator = Truncator(truncated_length="max", error_on_short=False)
    truncator.fit(X)
    Xt = truncator.transform(X2)

    assert isinstance(Xt, list)
    assert len(Xt) == len(X2)
    assert len(Xt[0].shape) == 2 and Xt[0].shape[0] == 1
    assert all(
        (
            Xt[i].shape[1] == max_length
            if X2[i].shape[1] > max_length
            else Xt[i].shape[1] == X2[i].shape[1]
        )
        for i in range(len(Xt))
    )

    X3, _ = make_example_3d_numpy_list(min_n_timepoints=4, max_n_timepoints=7)
    Xt2 = truncator.transform(X3)
    assert deep_equals(Xt2, X3)


def test_incorrect_arguments():
    """Test Truncator with incorrect constructor arguments."""
    X, _ = make_example_3d_numpy()

    truncator = Truncator(truncated_length="invalid")
    with pytest.raises(ValueError, match="truncated_length must be"):
        truncator.fit_transform(X)

    truncator = Truncator(truncated_length=20)
    with pytest.raises(ValueError, match="less than the provided truncated_length"):
        truncator.fit_transform(X)
