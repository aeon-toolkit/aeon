"""Test for Bounding Matrix."""

__author__ = ["chrisholder"]
__maintainer__ = []

import numpy as np

from aeon.distances import create_bounding_matrix


def test_full_bounding():
    """Test function to check the creation of a full bounding matrix."""
    matrix = create_bounding_matrix(10, 10)
    assert np.all(matrix)


def test_window_bounding():
    """Test function to check the creation of a windowed bounding matrix."""
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


def test_itakura_parallelogram():
    """
    Test function to check the creation of an,
    Itakura parallelogram bounding matrix.
    """
    matrix = create_bounding_matrix(10, 10, itakura_max_slope=0.2)
    assert isinstance(matrix, np.ndarray)
