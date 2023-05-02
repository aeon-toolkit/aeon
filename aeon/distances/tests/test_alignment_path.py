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
        alignment_path,
):
    alignment_path_result = alignment_path(x, y)

    assert isinstance(alignment_path_result, tuple)
    assert isinstance(alignment_path_result[0], list)
    assert isinstance(alignment_path_result[1], float)

    distance_result = distance(x, y)
    assert_almost_equal(alignment_path_result[1], distance_result)


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    if "alignment_path" not in dist:
        return

    # Test univariate
    if dist["name"] != "ddtw" and dist["name"] != "wddtw" and dist["name"] != "lcss":
        _validate_cost_matrix_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )

    _validate_cost_matrix_result(
        create_test_distance_numpy(10),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(2, 1, 10)[0],
        create_test_distance_numpy(2, 1, 10, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test multivariate
    _validate_cost_matrix_result(
        create_test_distance_numpy(10, 10),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    if dist["name"] != "lcss":
        _validate_cost_matrix_result(
            create_test_distance_numpy(10, 10, 10),
            create_test_distance_numpy(10, 10, 10, random_state=2),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )

    # Test unequal length
    _validate_cost_matrix_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    _validate_cost_matrix_result(
        create_test_distance_numpy(10, 5),
        create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    if dist["name"] != "lcss":
        _validate_cost_matrix_result(
            create_test_distance_numpy(10, 10, 5),
            create_test_distance_numpy(10, 10, 10, random_state=2),
            dist["name"],
            dist["distance"],
            dist["alignment_path"],
        )
