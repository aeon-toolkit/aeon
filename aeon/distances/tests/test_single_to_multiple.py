import numpy as np
import pytest
from aeon.distances.tests.test_new_distances import DISTANCES
from aeon.distances.tests._utils import (
    create_test_distance_numpy
)


def _validate_single_to_multiple_result(
        x,
        y,
        name,  # This will be used in a later pr
        distance,
        single_to_multiple_distance,
):
    # distance = _test_generated_jit_distance_function(distance)
    # single_to_multiple_distance = _test_generated_jit_distance_function(single_to_multiple_distance)

    single_to_multiple_result = single_to_multiple_distance(x, y)

    expected_size = len(y)

    if y.ndim == 1:
        expected_size = 1

    assert isinstance(single_to_multiple_result, np.ndarray)
    assert single_to_multiple_result.shape[0] == expected_size

    for i in range(len(y)):
        curr_y = y[i]
        curr = single_to_multiple_result[i]
        dist = distance(x, curr_y)
        assert dist == curr


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_single_to_multiple_distances(dist):
    _validate_single_to_multiple_result(
        np.array([10.0]),
        np.array([[15.0]]),
        dist["name"],
        dist["distance"],
        dist["single_to_multiple_distance"],
    )

    _validate_single_to_multiple_result(
        create_test_distance_numpy(5, 5),
        create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["single_to_multiple_distance"],
    )
