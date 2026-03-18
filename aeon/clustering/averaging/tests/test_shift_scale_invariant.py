"""Tests for shift-invariant averaging."""

import numpy as np
import pytest

from aeon.clustering.averaging import shift_invariant_average
from aeon.clustering.averaging._averaging import _resolve_average_callable
from aeon.datasets import load_gunpoint
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.expected_results.expected_average_results import (
    expected_shift_invariant_multi,
    expected_shift_invariant_uni,
    expected_shift_with_params,
)
from aeon.testing.testing_config import MULTITHREAD_TESTING


def test_univariate_shift_invariant_average():
    """Test univariate shift-invariant averaging."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    avg = shift_invariant_average(data)
    other_avg = _resolve_average_callable("shift_scale")(data)

    assert avg.shape == (1, 10)
    assert np.allclose(avg, expected_shift_invariant_uni)
    assert np.array_equal(avg, other_avg)


def test_multivariate_shift_invariant_average():
    """Test multivariate shift-invariant averaging."""
    data = make_example_3d_numpy(20, 3, 10, return_y=False, random_state=1)
    avg = shift_invariant_average(data)
    other_avg = _resolve_average_callable("shift_scale")(data)

    assert avg.shape == (3, 10)
    assert np.allclose(avg, expected_shift_invariant_multi)
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
    assert np.allclose(avg, expected_shift_with_params)
    assert not np.array_equal(avg, original)


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_shift_scale_threaded(n_jobs):
    """Test subgradient threaded functionality."""
    data = make_example_3d_numpy(10, 3, 10, random_state=2, return_y=False)
    serial = shift_invariant_average(data, n_jobs=1)
    parallel = shift_invariant_average(data, n_jobs=n_jobs)
    assert serial.shape == parallel.shape
    assert np.allclose(serial, parallel)
