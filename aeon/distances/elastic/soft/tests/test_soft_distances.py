"""Test soft distances."""

import numpy as np
import pytest

from aeon.distances._distance import DISTANCES_DICT, SOFT_DISTANCES, distance
from aeon.testing.data_generation import make_example_2d_numpy_series


def _test_gradient_func(dist, x, y):
    gradient_func = DISTANCES_DICT[dist]["gradient"]
    for gamma in [1.0, 0.1, 0.001]:
        curr_grad, curr_dist = gradient_func(x, y, gamma=gamma)
        assert isinstance(curr_grad, np.ndarray)
        assert curr_grad.shape == (x.shape[1], y.shape[1])
        assert curr_grad[-1, -1] == 1.0

        dist_func_dist = distance(x, y, method=dist, gamma=gamma)
        assert np.isclose(dist_func_dist, abs(curr_dist), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dist", SOFT_DISTANCES)
def test_gradient(dist):
    """Test for gradient for various distances."""
    univariate_x = make_example_2d_numpy_series(10, 1, random_state=1)
    univariate_y = make_example_2d_numpy_series(10, 1, random_state=2)

    _test_gradient_func(dist, univariate_x, univariate_y)

    multivariate_x = make_example_2d_numpy_series(10, 10, random_state=1)
    multivariate_y = make_example_2d_numpy_series(10, 10, random_state=2)

    _test_gradient_func(dist, multivariate_x, multivariate_y)
