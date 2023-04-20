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


@pytest.mark.parametrize("dist", DISTANCES)
def test_pairwise_distance(dist):
    _validate_pairwise_result(
        create_test_distance_numpy(5, 5),
        dist["name"],
        dist["distance"],
        dist["pairwise_distance"],
    )

    # _validate_pairwise_result(
    #     create_test_distance_numpy(5, 1, 5),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
    #
    # _validate_pairwise_result(
    #     create_test_distance_numpy(5, 5, 5),
    #     dist["name"],
    #     dist["distance"],
    #     dist["pairwise_distance"],
    # )
