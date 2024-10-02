"""Test the miscellaneous distance functions."""

import numpy as np

from aeon.distances import (
    shift_scale_invariant_best_shift,
    shift_scale_invariant_distance,
)
from aeon.testing.data_generation import make_example_2d_numpy_series


def test_shift_scale_invariant_distance():
    """Test the shift-scale invariant distance shift function."""
    univ_x = make_example_2d_numpy_series(n_channels=1, n_timepoints=10, random_state=1)
    univ_y = make_example_2d_numpy_series(n_channels=1, n_timepoints=10, random_state=2)

    multi_x = make_example_2d_numpy_series(
        n_channels=3, n_timepoints=10, random_state=1
    )
    multi_y = make_example_2d_numpy_series(
        n_channels=3, n_timepoints=10, random_state=2
    )

    univariate_dist = shift_scale_invariant_distance(univ_x, univ_y)
    multivariate_dist = shift_scale_invariant_distance(multi_x, multi_y)

    univariate_shift = shift_scale_invariant_best_shift(univ_x, univ_y)
    multivariate_shift = shift_scale_invariant_best_shift(multi_x, multi_y)

    assert univariate_dist == univariate_shift[0]
    assert multivariate_dist == multivariate_shift[0]

    assert isinstance(univariate_shift[1], np.ndarray)
    assert univariate_shift[1].shape == (10,)
    assert isinstance(multivariate_shift[1], np.ndarray)
    assert multivariate_shift[1].shape == (3, 10)

    # x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [0, 1, 0, 2, 0]])
    # y = np.array([[11, 12, 13, 14, 15], [3, 22, 5, 4, 11], [12, 3, 4, 5, 19]])
    # assert shift_scale_invariant_distance(x, y) == 1.269141579294335
