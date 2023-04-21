import numpy as np
import pytest

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


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    if "cost_matrix" not in dist:
        return

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

