"""Tests for shift-invariant averaging."""

import numpy as np

from aeon.clustering.averaging import shift_invariant_average
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.datasets import load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy


def test_univariate_shift_invariant_average():
    """Test univariate shift-invariant averaging."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    avg = shift_invariant_average(data)
    other_avg = _resolve_average_callable("shift_scale")(data)

    assert avg.shape == (1, 10)
    assert np.array_equal(avg, other_avg)


def test_multivariate_shift_invariant_average():
    """Test multivariate shift-invariant averaging."""
    data = make_example_3d_numpy(20, 3, 10, return_y=False, random_state=1)
    avg = shift_invariant_average(data)
    other_avg = _resolve_average_callable("shift_scale")(data)

    assert avg.shape == (3, 10)
    assert np.array_equal(avg, other_avg)


def test_shift_invariant_average_with_initial_center():
    """Tset shift-invariant averaging with initial center."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    original = shift_invariant_average(data)
    avg = shift_invariant_average(data, initial_center=np.ones((1, 10)))
    other_avg = _resolve_average_callable("shift_scale")(
        data, initial_center=np.ones((1, 10))
    )

    assert avg.shape == (1, 10)
    assert np.array_equal(avg, other_avg)
    assert not np.array_equal(avg, original)


def test_shift_invariant_average_with_max_shift():
    """Test shift-invariant averaging with different max_shift."""
    data, y_train = load_gunpoint(split="train")
    # get gunpoint data with class 1
    data = data[y_train == "1"]
    original = shift_invariant_average(data)
    avg = shift_invariant_average(data, max_shift=2)
    other_avg = _resolve_average_callable("shift_scale")(data, max_shift=2)

    assert avg.shape == data[0].shape
    assert np.array_equal(avg, other_avg)
    assert not np.array_equal(avg, original)
