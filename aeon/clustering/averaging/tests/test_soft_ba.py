"""Test soft ba."""

import warnings

import pytest

from aeon.clustering.averaging._soft_barycentre import soft_barycenter_average
from aeon.distances._distance import SOFT_DISTANCES
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize("dist", SOFT_DISTANCES)
def test_no_convergence_warning(dist):
    """Test that no convergence warning is raised."""
    X = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")

        soft_barycenter_average(X, distance=dist, gamma=1.0, tol=1e-5, max_iters=50)

        for warning in recorded_warnings:
            # This warning means the gradient doesnt match the expected value
            assert "ABNORMAL_TERMINATION_IN_LNSRCH" not in str(warning.message)
            assert "Optimisation failed to converge" not in str(warning.message)
