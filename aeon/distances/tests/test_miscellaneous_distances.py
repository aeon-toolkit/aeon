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


def test_shift_scale_invariant_distance_float32():
    """Test that float32 input does not raise a numba TypingError (#3722).

    The zero-padding in the shift branches used to default to float64, which could
    not unify with the float32 input inside the numba-compiled distance.
    """
    x = make_example_2d_numpy_series(n_channels=1, n_timepoints=10, random_state=1)
    y = make_example_2d_numpy_series(n_channels=1, n_timepoints=10, random_state=2)

    dist64 = shift_scale_invariant_distance(x.astype(np.float64), y.astype(np.float64))
    dist32 = shift_scale_invariant_distance(x.astype(np.float32), y.astype(np.float32))

    assert_almost_equal(dist32, dist64, decimal=4)

    # A float32 input must also yield a float32 shifted output: the zero-padding
    # now matches the input dtype instead of forcing float64 (#3722).
    _, shifted32 = shift_scale_invariant_best_shift(
        x.astype(np.float32), y.astype(np.float32)
    )
    assert shifted32.dtype == np.float32


def test_minkowski_pairwise_weights_match_single_series():
    """Weights must be series-shaped and applied uniformly to every pair (#3861).

    Previously the pairwise helpers indexed w by case, causing out-of-bounds
    reads, crashes on multivariate input, and asymmetric X-to-y vs y-to-X.
    """
    from aeon.distances import minkowski_distance, minkowski_pairwise_distance

    rng = np.random.default_rng(0)

    # Univariate collection, series-shaped weights
    X = rng.normal(size=(4, 1, 5))
    Y = rng.normal(size=(3, 1, 5))
    w = np.abs(rng.normal(size=(1, 5)))

    expected_self = np.array(
        [[minkowski_distance(a, b, p=2, w=w) for b in X] for a in X]
    )
    expected_xy = np.array(
        [[minkowski_distance(a, b, p=2, w=w) for b in Y] for a in X]
    )

    got_self = minkowski_pairwise_distance(X, p=2, w=w)
    got_xy = minkowski_pairwise_distance(X, Y, p=2, w=w)

    assert_almost_equal(got_self, expected_self)
    assert_almost_equal(got_xy, expected_xy)
    # Symmetry under swap of arguments
    assert_almost_equal(
        minkowski_pairwise_distance(Y, X, p=2, w=w), expected_xy.T
    )

    # Multivariate collection
    Xm = rng.normal(size=(3, 2, 5))
    wm = np.abs(rng.normal(size=(2, 5)))
    expected_m = np.array(
        [[minkowski_distance(a, b, p=2, w=wm) for b in Xm] for a in Xm]
    )
    got_m = minkowski_pairwise_distance(Xm, p=2, w=wm)
    assert_almost_equal(got_m, expected_m)

    # Wrong weight shape must raise clearly
    with np.testing.assert_raises(ValueError):
        minkowski_pairwise_distance(X, p=2, w=np.ones(5))
