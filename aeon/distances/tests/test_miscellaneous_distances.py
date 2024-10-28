"""Test the miscellaneous distance functions."""

import numpy as np
from numpy.ma.testutils import assert_almost_equal

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

    assert_almost_equal(univariate_dist, univariate_shift[0])
    assert_almost_equal(multivariate_dist, multivariate_shift[0])

    assert isinstance(univariate_shift[1], np.ndarray)
    assert univariate_shift[1].shape == (10,)
    assert isinstance(multivariate_shift[1], np.ndarray)
    assert multivariate_shift[1].shape == (3, 10)
