"""Tests for averaging helpers in ``_averaging`` and ``_barycenter_averaging``."""

import numpy as np
import pytest

from aeon.clustering.averaging import elastic_barycenter_average, mean_average
from aeon.clustering.averaging._averaging import (
    _AVERAGE_DICT,
    _resolve_average_callable,
)
from aeon.testing.data_generation import make_example_3d_numpy


def test_mean_average_multiple():
    """Test mean_average returns the per-timepoint mean for multiple cases."""
    X = make_example_3d_numpy(5, 2, 8, return_y=False, random_state=1)
    avg = mean_average(X)
    assert avg.shape == (2, 8)
    assert np.allclose(avg, X.mean(axis=0))


def test_mean_average_single_case():
    """Test mean_average returns the input unchanged for a single case."""
    X = make_example_3d_numpy(1, 1, 8, return_y=False, random_state=1)
    avg = mean_average(X)
    assert np.array_equal(avg, X)


def test_resolve_average_callable_string():
    """Test resolving every valid averaging string returns the right callable."""
    for key, expected in _AVERAGE_DICT.items():
        assert _resolve_average_callable(key) is expected


def test_resolve_average_callable_passthrough():
    """Test a callable is returned unchanged."""

    def custom(X):
        return X.mean(axis=0)

    assert _resolve_average_callable(custom) is custom


def test_resolve_average_callable_invalid():
    """Test an invalid averaging string raises a ValueError."""
    with pytest.raises(ValueError, match="averaging_method string is invalid"):
        _resolve_average_callable("not_a_real_method")


def test_elastic_barycenter_average_invalid_method():
    """Test elastic_barycenter_average rejects an unknown method."""
    X = make_example_3d_numpy(5, 1, 8, return_y=False, random_state=1)
    with pytest.raises(ValueError, match="Invalid method"):
        elastic_barycenter_average(X, method="not_a_real_method")
