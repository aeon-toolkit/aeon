"""Tests for KShape clustering."""

import numpy as np
import pytest

from aeon.clustering import KShape
from aeon.clustering._k_shape import (
    _ncc_time_major,
    _sbd_align_1d,
    _shape_extraction,
    _shift_zeropad_1d,
)
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize("n_channels", [1, 2])
def test_kshape_fit_structure(n_channels):
    """Fitted KShape produces a valid partition, centres and diagnostics."""
    n_cases, n_timepoints, n_clusters = 12, 12, 2
    X = make_example_3d_numpy(
        n_cases, n_channels, n_timepoints, random_state=1, return_y=False
    )

    ks = KShape(n_clusters=n_clusters, random_state=1)
    ks.fit(X)

    assert ks.labels_.shape == (n_cases,)
    assert np.issubdtype(ks.labels_.dtype, np.integer)
    assert set(np.unique(ks.labels_)) <= set(range(n_clusters))

    assert ks.cluster_centres_.shape == (n_clusters, n_channels, n_timepoints)
    assert np.all(np.isfinite(ks.cluster_centres_))

    assert np.isfinite(ks.inertia_) and ks.inertia_ >= 0
    assert ks.n_iter_ >= 1

    preds = ks.predict(X)
    assert preds.shape == (n_cases,)
    assert set(np.unique(preds)) <= set(range(n_clusters))


def test_kshape_no_z_normalise_and_multiple_inits():
    """z_normalise=False and n_init > 1 both fit and predict cleanly."""
    X = make_example_3d_numpy(10, 1, 10, random_state=2, return_y=False)
    ks = KShape(n_clusters=2, z_normalise=False, n_init=2, random_state=2)
    ks.fit(X)
    preds = ks.predict(X)
    assert preds.shape == (10,)
    assert np.isfinite(ks.inertia_)


def test_kshape_zero_and_array_init():
    """'zero' and explicit ndarray initial centres are both accepted."""
    X = make_example_3d_numpy(10, 1, 10, random_state=3, return_y=False)

    ks_zero = KShape(n_clusters=2, init="zero", random_state=3).fit(X)
    assert ks_zero.cluster_centres_.shape == (2, 1, 10)

    init = X[:2].copy()
    ks_arr = KShape(n_clusters=2, init=init, random_state=3).fit(X)
    assert ks_arr.cluster_centres_.shape == (2, 1, 10)


def test_kshape_param_validation():
    """Invalid constructor parameters raise at fit time."""
    X = make_example_3d_numpy(6, 1, 8, random_state=0, return_y=False)

    with pytest.raises(ValueError, match="n_clusters must be a positive integer"):
        KShape(n_clusters=0).fit(X)
    with pytest.raises(ValueError, match="cannot be larger than"):
        KShape(n_clusters=10).fit(X)
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        KShape(n_clusters=2, max_iter=0).fit(X)
    with pytest.raises(ValueError, match="n_init must be a positive integer"):
        KShape(n_clusters=2, n_init=0).fit(X)
    with pytest.raises(ValueError, match="tol must be non-negative"):
        KShape(n_clusters=2, tol=-1.0).fit(X)
    with pytest.raises(ValueError, match="init must be 'random', 'zero'"):
        KShape(n_clusters=2, init="bad").fit(X)
    with pytest.raises(ValueError, match="If init is an array"):
        KShape(n_clusters=2, init=np.zeros((3, 1, 8))).fit(X)


def test_kshape_verbose_max_iter_and_tol(capsys):
    """Verbose printing, the max_iter exit, and the tol-based exit."""
    X = make_example_3d_numpy(10, 1, 10, random_state=4, return_y=False)

    ks = KShape(n_clusters=2, max_iter=1, verbose=True, random_state=4).fit(X)
    assert ks.n_iter_ == 1
    assert "iter=1" in capsys.readouterr().out

    # this configuration needs six iterations to reach label stability, so
    # a huge tolerance must stop it early via the inertia criterion instead
    X_big = make_example_3d_numpy(24, 1, 16, random_state=2, return_y=False)
    baseline = KShape(n_clusters=3, max_iter=50, random_state=2).fit(X_big)
    ks_tol = KShape(n_clusters=3, tol=1e12, max_iter=50, random_state=2).fit(X_big)
    assert ks_tol.n_iter_ < baseline.n_iter_


def test_kshape_empty_cluster_reseeded():
    """Empty clusters are reseeded with a random training case."""
    X = make_example_3d_numpy(5, 1, 8, random_state=5, return_y=False)
    # more clusters than distinct random initial labels makes an empty
    # cluster very likely; the fit must still produce valid output
    ks = KShape(n_clusters=4, random_state=3, max_iter=5).fit(X)
    assert ks.labels_.shape == (5,)
    assert ks.cluster_centres_.shape == (4, 1, 8)
    assert np.all(np.isfinite(ks.cluster_centres_))


def test_shift_zeropad_1d():
    """Zero-padded shifting covers all shift regimes."""
    x = np.arange(1.0, 5.0)
    np.testing.assert_array_equal(_shift_zeropad_1d(x, 0), x)
    np.testing.assert_array_equal(_shift_zeropad_1d(x, 2), [0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(_shift_zeropad_1d(x, -2), [3.0, 4.0, 0.0, 0.0])
    # shifts of at least the length blank the series entirely
    np.testing.assert_array_equal(_shift_zeropad_1d(x, 5), np.zeros(4))
    np.testing.assert_array_equal(_shift_zeropad_1d(x, -5), np.zeros(4))


def test_ncc_time_major_zero_norm():
    """Zero-norm inputs return an all-zero correlation."""
    out = _ncc_time_major(np.zeros((6, 1)), np.zeros((6, 1)))
    assert out.shape == (11,)
    assert np.all(out == 0.0)


def test_sbd_align_1d_zero_centre():
    """A zero centre leaves the series unaligned."""
    x = np.arange(5.0)
    np.testing.assert_array_equal(_sbd_align_1d(np.zeros(5), x), x)


def test_shape_extraction_input_validation():
    """Shape extraction rejects empty and non-2D input."""
    P = np.eye(4) - 0.25 * np.ones((4, 4))
    with pytest.raises(ValueError, match="cannot be empty"):
        _shape_extraction(np.empty((0, 4)), P)
    with pytest.raises(ValueError, match="must be 2D"):
        _shape_extraction(np.zeros(4), P)
