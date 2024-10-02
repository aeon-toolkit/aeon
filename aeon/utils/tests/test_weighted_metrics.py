"""Test weighted metric."""

import numpy as np

from aeon.utils._weighted_metrics import weighted_geometric_mean


def test_weighted_geometric_mean():
    """Test weighted_geometric_mean."""
    y = np.array([1.0, 2.0, 3.0])
    weighted_geometric_mean(y, weights=np.array([0.1, 0.8, 0.1]))
