"""Tests for cost matrix."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import cost_matrix as compute_cost_matrix
from aeon.distances._distance import (
    DISTANCES,
    DISTANCES_DICT,
    SINGLE_POINT_NOT_SUPPORTED_DISTANCES,
)
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
)


def _validate_cost_matrix_result(
    x: np.ndarray,
    y: np.ndarray,
    name,
    distance,
    cost_matrix,
    check_xy_permuted=True,
):
    """Validate the result of the cost matrix function.

    Parameters
    ----------
    x (np.ndarray): The first input array.
    y (np.ndarray): The second input array.
    name: The name of the distance method.
    distance: The distance method function.
    cost_matrix: The cost matrix function.
    """
    original_x = x.copy()
    original_y = y.copy()
    cost_matrix_result = cost_matrix(x, y)
    cost_matrix_callable_result = DISTANCES_DICT[name]["cost_matrix"](x, y)

    assert isinstance(cost_matrix_result, np.ndarray)
    assert_almost_equal(cost_matrix_result, compute_cost_matrix(x, y, method=name))
    assert_almost_equal(cost_matrix_callable_result, cost_matrix_result)
    if name == "ddtw" or name == "wddtw":
        assert cost_matrix_result.shape == (x.shape[-1] - 2, y.shape[-1] - 2)
    elif name == "lcss":
        # lcss cm is one larger than the input
        assert cost_matrix_result.shape == (x.shape[-1] + 1, y.shape[-1] + 1)
    else:
        assert cost_matrix_result.shape == (x.shape[-1], y.shape[-1])

    distance_result = distance(x, y)

    if name == "lcss":
        if x.ndim != 3:
            curr_distance = 1 - (
                float(cost_matrix_result[-1, -1] / min(x.shape[-1], y.shape[-1]))
            )
            assert_almost_equal(curr_distance, distance_result)
    elif name == "edr":
        if x.ndim != 3:
            curr_distance = float(
                cost_matrix_result[-1, -1] / max(x.shape[-1], y.shape[-1])
            )
            assert_almost_equal(curr_distance, distance_result)
    elif name == "soft_dtw":
        assert_almost_equal(abs(cost_matrix_result[-1, -1]), distance_result)
    else:
        assert_almost_equal(cost_matrix_result[-1, -1], distance_result)

    # If unequal length swap where x and y are to ensure it works both ways around
    if original_x.shape[-1] != original_y.shape[-1] and check_xy_permuted:
        _validate_cost_matrix_result(
            original_y,
            original_x,
            name,
            distance,
            cost_matrix,
            check_xy_permuted=False,
        )


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    """Test for cost matrix for various distances."""
    if dist["name"] == "shape_dtw":
        return

    if "cost_matrix" not in dist:
        return

    # ================== Test equal length ==================
    # Test univariate of shape (n_timepoints,)
    _validate_cost_matrix_result(
        make_example_1d_numpy(10, random_state=1),
        make_example_1d_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # Test univariate of shape (1, n_timepoints)
    _validate_cost_matrix_result(
        make_example_2d_numpy_series(10, 1, random_state=1),
        make_example_2d_numpy_series(10, 1, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # Test multivariate of shape (n_channels, n_timepoints)
    _validate_cost_matrix_result(
        make_example_2d_numpy_series(10, 10, random_state=1),
        make_example_2d_numpy_series(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # ================== Test unequal length ==================
    # Test univariate unequal length of shape (n_timepoints,)
    _validate_cost_matrix_result(
        make_example_1d_numpy(5, random_state=1),
        make_example_1d_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # Test univariate unequal length of shape (1, n_timepoints)
    _validate_cost_matrix_result(
        make_example_2d_numpy_series(5, 1, random_state=1),
        make_example_2d_numpy_series(10, 1, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # Test multivariate unequal length of shape (n_channels, n_timepoints)
    _validate_cost_matrix_result(
        make_example_2d_numpy_series(5, 10, random_state=1),
        make_example_2d_numpy_series(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # ============== Test single point series ==============
    if dist["name"] not in SINGLE_POINT_NOT_SUPPORTED_DISTANCES:
        # Test singe point univariate of shape (1,)
        _validate_cost_matrix_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["cost_matrix"],
        )

        # Test singe point univariate of shape (1, 1)
        _validate_cost_matrix_result(
            np.array([[10.0]]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["cost_matrix"],
        )
