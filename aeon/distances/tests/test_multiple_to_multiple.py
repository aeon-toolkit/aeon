import numpy as np
import pytest
from aeon.distances.tests.test_new_distances import DISTANCES
from aeon.distances.tests._utils import _make_3d_series, create_test_distance_numpy


def _validate_multiple_to_multiple_result(
        x,
        y,
        name,  # This will be used in a later pr
        distance,
        multiple_to_multiple_distance,
):
    multiple_to_multiple_result = multiple_to_multiple_distance(x, y)

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


@pytest.mark.parametrize("dist", DISTANCES)
def test_new_multiple_to_multiple_distances(dist):
    _validate_multiple_to_multiple_result(
        np.array([10.0]),
        np.array([15.0]),
        dist["name"],
        dist["distance"],
        dist["multiple_to_multiple_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5),
        create_test_distance_numpy(5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["multiple_to_multiple_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 5),
        create_test_distance_numpy(5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["multiple_to_multiple_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 1, 5),
        create_test_distance_numpy(5, 1, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["multiple_to_multiple_distance"],
    )

    _validate_multiple_to_multiple_result(
        create_test_distance_numpy(5, 5, 5),
        create_test_distance_numpy(5, 5, 5, random_state=2),
        dist["name"],
        dist["distance"],
        dist["multiple_to_multiple_distance"],
    )
