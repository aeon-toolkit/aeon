# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

import numpy as np

from aeon.distances import create_bounding_matrix


def test_full_bounding():
    matrix = create_bounding_matrix(10, 10)
    assert np.all(matrix)


def test_window_bounding():
    matrix = create_bounding_matrix(5, 5, window=0.2)
    num_true = 0
    num_false = 0
    for row in matrix:
        for val in row:
            if val:
                num_true += 1
            else:
                num_false += 1

    assert num_true == 80
    assert num_false == 20
