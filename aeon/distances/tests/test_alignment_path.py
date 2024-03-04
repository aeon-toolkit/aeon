"""Test for Path Alignment."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import alignment_path as compute_alignment_path
from aeon.distances._distance import DISTANCES
from aeon.distances.tests.test_utils import _create_test_distance_numpy


def _validate_cost_matrix_result(
    x: np.ndarray,
    y: np.ndarray,
    name,
    distance,
    alignment_path,
):
    alignment_path_result = alignment_path(x, y)

    assert isinstance(alignment_path_result, tuple)
    assert isinstance(alignment_path_result[0], list)
    assert isinstance(alignment_path_result[1], float)
    assert compute_alignment_path(x, y, metric=name) == alignment_path_result

    distance_result = distance(x, y)
    assert_almost_equal(alignment_path_result[1], distance_result)


@pytest.mark.parametrize("dist", DISTANCES)
def test_cost_matrix(dist):
    """
    Test function to check the cost matrix for various distances.

    Parameters
    ----------
        dist(dict): A dictionary containing the details of the distances.
    """
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
        _create_test_distance_numpy(10),
        _create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test multivariate
    _validate_cost_matrix_result(
        _create_test_distance_numpy(10, 10),
        _create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    # Test unequal length
    _validate_cost_matrix_result(
        _create_test_distance_numpy(5),
        _create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )

    _validate_cost_matrix_result(
        _create_test_distance_numpy(10, 5),
        _create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        dist["alignment_path"],
    )
