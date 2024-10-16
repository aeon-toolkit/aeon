"""Test weighted metric."""

import numpy as np
import pytest

from aeon.utils.weighted_metrics import weighted_geometric_mean


def test_weighted_geometric_mean():
    """Test weighted_geometric_mean."""
    y = np.array([1.0, 2.0, 3.0])
    w = np.array([0.1, 0.8, 0.1])
    w2 = np.array([[0.1, 0.8, 0.1]]).T
    res = weighted_geometric_mean(y, w)
    assert round(res, 5) == 1.94328
    res2 = weighted_geometric_mean(y, w, axis=0)
    assert res == res2
    y2 = np.array([[1.0, 2.0, 3.0]]).T
    with pytest.raises(ValueError, match="do not match"):
        weighted_geometric_mean(y2, w, axis=1)
    weighted_geometric_mean(y2, w2, axis=1)
    with pytest.raises(
        ValueError, match="Input data and weights have inconsistent shapes"
    ):
        weighted_geometric_mean(y, w2)
