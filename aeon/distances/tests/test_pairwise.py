# -*- coding: utf-8 -*-
import numpy as np
import pytest

from aeon.distances.tests._utils import _make_3d_series, create_test_distance_numpy
from aeon.distances.tests.test_new_distances import DISTANCES


def _validate_pairwise_result(
        x: np.ndarray,
        name,  # This will be used in a later pr
        distance,
        pairwise_distance,
):
    multiple_to_multiple_result = pairwise_distance(x)

    expected_size = (len(x), len(x))

    assert isinstance(multiple_to_multiple_result, np.ndarray)
    assert multiple_to_multiple_result.shape == expected_size

    x = _make_3d_series(x)

    matrix = np.zeros((len(x), len(x)))

    for i in range(len(x)):
        curr_x = x[i]
        for j in range(len(x)):
            curr_y = x[j]
            matrix[i, j] = distance(curr_x, curr_y)

    assert np.allclose(matrix, multiple_to_multiple_result)


def _validate_multiple_to_multiple_result(
        x,
        y,
        name,  # This will be used in a later pr
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
        name,  # This will be used in a later pr
        distance,
        single_to_multiple_distance,
):
    single_to_multiple_result = single_to_multiple_distance(x, y)

    test = single_to_multiple_result.shape

    expected_size = len(y)
    if y.ndim == 1:
        expected_size = 1

    assert isinstance(single_to_multiple_result, np.ndarray)
    assert single_to_multiple_result.shape[-1] == expected_size

    for i in range(single_to_multiple_result.shape[-1]):
        curr_y = y[i]
        curr = single_to_multiple_result[0, i]
        dist = distance(x, curr_y)
        assert dist == curr


@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance(dist):
    _validate_pairwise_result(
        create_test_distance_numpy(5, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_pairwise_result(
        create_test_distance_numpy(5, 1, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_pairwise_result(
        create_test_distance_numpy(5, 5, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )


@pytest.mark.parametrize("dist", DISTANCES)
def test_multiple_to_multiple_distances(dist):
    # Univariate tests
    _validate_multiple_to_multiple_result(
        np.array([10.0]),
        np.array([15.0]),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Multivariate tests
    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 5),
        create_test_distance_numpy(5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Dataset tests
    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 1, 5),
        create_test_distance_numpy(5, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 5, 5),
        create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Unequal length tests
    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(2, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 5),
        create_test_distance_numpy(5, 2, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 5, 5),
        create_test_distance_numpy(5, 5, 2, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )
new_distance = ["euclidean", "dtw"]

@pytest.mark.parametrize("dist", DISTANCES)
def test_new_single_to_multiple_distances(dist):
    # Univariate tests
    _validate_single_to_multiple_result(
        np.array([10.0]),
        np.array([[15.0]]),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(2, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        create_test_distance_numpy(2, 1, 5)[0],
        create_test_distance_numpy(5, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Multivariate tests
    _validate_single_to_multiple_result(
        create_test_distance_numpy(5, 5),
        create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # Unequal length tests
    _validate_single_to_multiple_result(
        create_test_distance_numpy(2),
        create_test_distance_numpy(2, 1, 5, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(2, 1, 2, random_state=2)[0],
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        create_test_distance_numpy(5, 2),
        create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    _validate_single_to_multiple_result(
        create_test_distance_numpy(5, 5),
        create_test_distance_numpy(5, 5, 2, random_state=2),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )
