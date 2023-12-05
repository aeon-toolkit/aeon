import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import alignment_path, cost_matrix
from aeon.distances import distance
from aeon.distances import distance as compute_distance
from aeon.distances import get_distance_function_names, pairwise_distance
from aeon.distances._distance import (
    DISTANCES,
    _custom_func_pairwise,
    _resolve_key_from_distance,
)
from aeon.distances.tests.test_expected_results import _expected_distance_results
from aeon.distances.tests.test_utils import _create_test_distance_numpy


def _validate_distance_result(x, y, name, distance, expected_result=10):
    if expected_result is None:
        return

    dist_result = distance(x, y)

    assert isinstance(dist_result, float)
    assert_almost_equal(dist_result, expected_result)
    assert_almost_equal(dist_result, compute_distance(x, y, metric=name))
    assert_almost_equal(dist_result, compute_distance(x, y, metric=distance))

    dist_result_to_self = distance(x, x)
    assert isinstance(dist_result_to_self, float)


@pytest.mark.parametrize("dist", DISTANCES)
def test_distances(dist):
    # Test univariate
    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
        _validate_distance_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            _expected_distance_results[dist["name"]][0],
        )

    _validate_distance_result(
        _create_test_distance_numpy(10),
        _create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][1],
    )

    # Test multivariate
    _validate_distance_result(
        _create_test_distance_numpy(10, 10),
        _create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][2],
    )

    # Test unequal length
    _validate_distance_result(
        _create_test_distance_numpy(5),
        _create_test_distance_numpy(10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][3],
    )

    _validate_distance_result(
        _create_test_distance_numpy(10, 5),
        _create_test_distance_numpy(10, 10, random_state=2),
        dist["name"],
        dist["distance"],
        _expected_distance_results[dist["name"]][4],
    )


def test_get_distance_function_names():
    assert get_distance_function_names() == sorted([dist["name"] for dist in DISTANCES])


def test__resolve_key_from_distance():
    with pytest.raises(ValueError, match="Unknown metric"):
        _resolve_key_from_distance(metric="FOO", key="cost_matrix")
    with pytest.raises(ValueError):
        _resolve_key_from_distance(metric="dtw", key="FOO")

    def foo(x, y):
        return 0

    assert callable(_resolve_key_from_distance(foo, key="FOO"))


def test_incorrect_inputs():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    with pytest.raises(
        ValueError, match="Metric must be one of the supported strings or a " "callable"
    ):
        distance(x, y, metric="FOO")
    with pytest.raises(
        ValueError, match="Metric must be one of the supported strings or a " "callable"
    ):
        pairwise_distance(x, y, metric="FOO")
    with pytest.raises(ValueError, match="Metric must be one of the supported strings"):
        alignment_path(x, y, metric="FOO")
    with pytest.raises(ValueError, match="Metric must be one of the supported strings"):
        cost_matrix(x, y, metric="FOO")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError, match="x and y must be 2D or 3D arrays"):
        _custom_func_pairwise(x)
