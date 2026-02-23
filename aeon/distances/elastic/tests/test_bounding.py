"""Test for bounding matrix."""

import numpy as np

from aeon.distances import create_bounding_matrix


def test_full_bounding():
    """Test to check the creation of a full bounding matrix."""
    matrix = create_bounding_matrix(10, 10)
    assert np.all(matrix)


def test_window_bounding():
    """Test to check the creation of a windowed bounding matrix."""
    matrix = create_bounding_matrix(10, 10, window=0.2)
    num_true = 0
    num_false = 0
    for row in matrix:
        for val in row:
            if val:
                num_true += 1
            else:
                num_false += 1

    assert num_true == 44
    assert num_false == 56

    unequal_1 = create_bounding_matrix(5, 10, window=0.2)
    unequal_2 = create_bounding_matrix(10, 5, window=0.2).T
    assert np.array_equal(unequal_2, unequal_1)


def test_itakura_parallelogram():
    """Test to check the creation of an Itakura parallelogram bounding matrix."""
    matrix = create_bounding_matrix(10, 10, itakura_max_slope=0.2)
    assert isinstance(matrix, np.ndarray)

    expected_result_5_7 = np.array(
        [
            [True, False, False, False, False, False, False],
            [False, True, True, True, True, False, False],
            [False, False, True, True, True, False, False],
            [False, False, True, True, True, True, False],
            [False, False, False, False, False, False, True],
        ]
    )

    expected_result_7_5 = np.array(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, True, False],
            [False, False, False, False, True],
        ]
    )

    matrix = create_bounding_matrix(5, 7, itakura_max_slope=0.5)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (5, 7)
    assert np.array_equal(matrix, expected_result_5_7)

    matrix = create_bounding_matrix(7, 5, itakura_max_slope=0.5)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (7, 5)
    assert np.array_equal(matrix, expected_result_7_5)
