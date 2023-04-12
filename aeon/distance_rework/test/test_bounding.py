import numpy as np
from aeon.distance_rework import create_bounding_matrix


def test_full_bounding():
    matrix = create_bounding_matrix(10, 10)
    assert np.all(np.isfinite(matrix))


def test_window_bounding():
    matrix = create_bounding_matrix(10, 10, window=0.5)
    num_finite = 0
    num_infinite = 0
    for row in matrix:
        for val in row:
            if np.isfinite(val):
                num_finite += 1
            else:
                num_infinite += 1

    assert num_finite == 75
    assert num_infinite == 25
