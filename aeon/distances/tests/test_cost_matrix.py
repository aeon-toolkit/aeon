import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.tests._utils import _make_3d_series, create_test_distance_numpy
from aeon.distances.tests.test_new_distances import DISTANCES


def _validate_cost_matrix_result(
        x: np.ndarray,
        y: np.ndarray,
        name,  # This will be used in a later pr
        distance,
        cost_matrix,
):
    cost_matrix_result = cost_matrix(x, y)

    assert isinstance(cost_matrix_result, np.ndarray)
    assert cost_matrix_result.shape == (x.shape[-1], y.shape[-1])

    distance_result = distance(x, y)

    assert_almost_equal(cost_matrix_result[-1, -1], distance_result)


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    if "cost_matrix" not in dist:
        return

    # Test univariate
    _validate_cost_matrix_result(
        np.array([10.0]),
        np.array([15.0]),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(10),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(2, 1, 10)[0],
        create_test_distance_numpy(2, 1, 10, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # Test multivariate
    _validate_cost_matrix_result(
        create_test_distance_numpy(10, 10),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(10, 10, 10),
        create_test_distance_numpy(10, 10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    # Test unequal length
    _validate_cost_matrix_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(10, 5),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(10, 10, 5),
        create_test_distance_numpy(10, 10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["cost_matrix"],
    )
