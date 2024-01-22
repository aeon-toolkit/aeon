import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from aeon.distances import pairwise_distance as compute_pairwise_distance
from aeon.distances._distance import DISTANCES
from aeon.distances.tests.test_utils import _create_test_distance_numpy, _make_3d_series


def _validate_pairwise_result(
    x: np.ndarray,
    name,
    distance,
    pairwise_distance,
):
    pairwise_result = pairwise_distance(x)

    expected_size = (len(x), len(x))

    assert isinstance(pairwise_result, np.ndarray)
    assert pairwise_result.shape == expected_size
    assert_almost_equal(pairwise_result, compute_pairwise_distance(x, metric=name))
    assert_almost_equal(pairwise_result, compute_pairwise_distance(x, metric=distance))

    x = _make_3d_series(x)

    matrix = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(x)):
            curr_y = x[j]
            matrix[i, j] = distance(curr_x, curr_y)

    assert np.allclose(matrix, pairwise_result)


def _validate_multiple_to_multiple_result(
    x,
    y,
    name,
    distance,
    multiple_to_multiple_distance,
):
    multiple_to_multiple_result = multiple_to_multiple_distance(x, y)

    if x.ndim == 1 and y.ndim == 1:
        expected_size = (1, 1)
    else:
        expected_size = (len(x), len(y))

    assert isinstance(multiple_to_multiple_result, np.ndarray)
    assert multiple_to_multiple_result.shape == expected_size

    assert_almost_equal(
        multiple_to_multiple_result, compute_pairwise_distance(x, y, metric=name)
    )
    assert_almost_equal(
        multiple_to_multiple_result,
        compute_pairwise_distance(x, y, metric=distance),
    )

    x = _make_3d_series(x)
    y = _make_3d_series(y)

    matrix = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(y)):
            curr_y = y[j]
            matrix[i, j] = distance(curr_x, curr_y)

    assert np.allclose(matrix, multiple_to_multiple_result)


def _validate_single_to_multiple_result(
    x,
    y,
    name,
    distance,
    single_to_multiple_distance,
):
    single_to_multiple_result = single_to_multiple_distance(x, y)

    expected_size = len(y)
    if y.ndim == 1:
        expected_size = 1

    assert isinstance(single_to_multiple_result, np.ndarray)
    assert single_to_multiple_result.shape[-1] == expected_size
    assert_almost_equal(
        single_to_multiple_result, compute_pairwise_distance(x, y, metric=name)
    )
    assert_almost_equal(
        single_to_multiple_result, compute_pairwise_distance(x, y, metric=distance)
    )

    for i in range(single_to_multiple_result.shape[-1]):
        curr_y = y[i]
        curr = single_to_multiple_result[0, i]
        dist = distance(x, curr_y)
        assert_almost_equal(dist, curr)


@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance(dist):
    """Test pairwise distance function."""
    _validate_pairwise_result(
        _create_test_distance_numpy(5, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_pairwise_result(
        _create_test_distance_numpy(5, 1, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_pairwise_result(
        _create_test_distance_numpy(5, 5, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )


@pytest.mark.parametrize("dist", DISTANCES)
def test_multiple_to_multiple_distances(dist):
    "Test multiple to multiple distances."
    # Univariate tests
    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
        _validate_multiple_to_multiple_result(
            np.array([10.0]),
            np.array([15.0]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5),
        _create_test_distance_numpy(5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Multivariate tests
    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5, 5),
        _create_test_distance_numpy(5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Dataset tests
    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5, 1, 5),
        _create_test_distance_numpy(5, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5, 5, 5),
        _create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Unequal length tests
    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5),
        _create_test_distance_numpy(3, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5, 5),
        _create_test_distance_numpy(5, 3, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        _create_test_distance_numpy(5, 5, 5),
        _create_test_distance_numpy(5, 5, 3, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )


new_distance = ["euclidean", "dtw"]


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_single_to_multiple_distances(dist):
    # Univariate tests

    if dist["name"] != "ddtw" and dist["name"] != "wddtw":
        _validate_single_to_multiple_result(
            np.array([10.0]),
            np.array([[15.0]]),
            dist["name"],
            dist["distance"],
            dist["pairwise_distance"],
        )

    _validate_single_to_multiple_result(
        _create_test_distance_numpy(5),
        _create_test_distance_numpy(3, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        _create_test_distance_numpy(3, 1, 5)[0],
        _create_test_distance_numpy(5, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Multivariate tests
    _validate_single_to_multiple_result(
        _create_test_distance_numpy(5, 5),
        _create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Unequal length tests
    _validate_single_to_multiple_result(
        _create_test_distance_numpy(3),
        _create_test_distance_numpy(3, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        _create_test_distance_numpy(5),
        _create_test_distance_numpy(3, 1, 3, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        _create_test_distance_numpy(5, 3),
        _create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        _create_test_distance_numpy(5, 5),
        _create_test_distance_numpy(5, 5, 3, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )
