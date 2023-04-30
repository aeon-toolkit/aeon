# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances.tests._utils import create_test_distance_numpy
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
            distance = 1 - float(
                cost_matrix_result[-1, -1] / min(x.shape[-1], y.shape[-1])
            )
            assert_almost_equal(distance, distance_result)
    else:
        assert_almost_equal(cost_matrix_result[-1, -1], distance_result)


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    if "cost_matrix" not in dist:
        return

    # Test univariate
    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
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
