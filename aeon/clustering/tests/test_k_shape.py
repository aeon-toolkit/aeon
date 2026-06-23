"""Tests for KShape clusterer."""

import numpy as np
import pytest

from aeon.clustering._k_shape import (
    KShape,
    _ncc_time_major,
    _sbd_align_1d,
    _shape_extraction,
    _shift_zeropad_1d,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.testing_config import MULTITHREAD_TESTING


def test_k_shape_univariate():
    """Test KShape with univariate data."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=2, random_state=1, max_iter=10)
    clusterer.fit(data)
    preds = clusterer.predict(data)

    assert clusterer.labels_.shape == (20,)
    assert len(set(clusterer.labels_)) == 2
    assert np.array_equal(clusterer.labels_, preds)
    assert clusterer.cluster_centres_.shape == (2, 1, 10)
    assert isinstance(clusterer.inertia_, float)
    assert isinstance(clusterer.n_iter_, int)
    assert clusterer.n_iter_ >= 1


def test_k_shape_multivariate():
    """Test KShape with multivariate data."""
    data = make_example_3d_numpy(15, 3, 12, return_y=False, random_state=2)
    clusterer = KShape(n_clusters=3, random_state=2, max_iter=5)
    clusterer.fit(data)
    preds = clusterer.predict(data)

    assert clusterer.labels_.shape == (15,)
    assert clusterer.cluster_centres_.shape == (3, 3, 12)
    assert preds.shape == (15,)
    assert set(preds).issubset({0, 1, 2})


def test_k_shape_deterministic():
    """Test KShape is deterministic given a random_state."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    first = KShape(n_clusters=2, random_state=1, max_iter=10).fit_predict(data)
    second = KShape(n_clusters=2, random_state=1, max_iter=10).fit_predict(data)
    assert np.array_equal(first, second)


def test_k_shape_init_zero():
    """Test KShape with zero centre initialisation."""
    data = make_example_3d_numpy(15, 2, 12, return_y=False, random_state=2)
    clusterer = KShape(n_clusters=3, init="zero", random_state=2, max_iter=5)
    clusterer.fit(data)
    assert clusterer.cluster_centres_.shape == (3, 2, 12)
    assert clusterer.labels_.shape == (15,)


def test_k_shape_init_array():
    """Test KShape with explicit array initialisation."""
    data = make_example_3d_numpy(10, 1, 10, return_y=False, random_state=3)
    init = make_example_3d_numpy(2, 1, 10, return_y=False, random_state=4)
    clusterer = KShape(n_clusters=2, init=init, random_state=3, max_iter=3)
    clusterer.fit(data)
    assert clusterer.cluster_centres_.shape == (2, 1, 10)
    # the original init array must not be mutated
    assert init.shape == (2, 1, 10)


def test_k_shape_no_z_normalise():
    """Test KShape without z-normalisation."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=2, z_normalise=False, random_state=1, max_iter=5)
    clusterer.fit(data)
    preds = clusterer.predict(data)
    assert clusterer.cluster_centres_.shape == (2, 1, 10)
    assert preds.shape == (20,)


def test_k_shape_n_init():
    """Test KShape keeps the best of several initialisations."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=2, n_init=3, random_state=1, max_iter=5)
    clusterer.fit(data)
    assert clusterer.labels_.shape == (20,)
    assert clusterer.inertia_ >= 0


def test_k_shape_predict_proba():
    """Test KShape predict_proba returns one-hot probabilities."""
    data = make_example_3d_numpy(15, 1, 10, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=3, random_state=1, max_iter=5)
    clusterer.fit(data)
    proba = clusterer.predict_proba(data)
    assert proba.shape == (15, 3)
    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1


def test_k_shape_verbose(capsys):
    """Test KShape prints debugging info when verbose is True."""
    data = make_example_3d_numpy(15, 1, 10, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=2, random_state=1, max_iter=3, verbose=True)
    clusterer.fit(data)
    captured = capsys.readouterr()
    assert "inertia" in captured.out


@pytest.mark.parametrize(
    "params,match",
    [
        ({"n_clusters": 0}, "n_clusters must be a positive integer"),
        ({"n_clusters": 2.5}, "n_clusters must be a positive integer"),
        ({"n_clusters": 100}, "cannot be larger than"),
        ({"n_clusters": 2, "max_iter": 0}, "max_iter must be a positive integer"),
        ({"n_clusters": 2, "n_init": 0}, "n_init must be a positive integer"),
        ({"n_clusters": 2, "tol": -1.0}, "tol must be non-negative"),
        ({"n_clusters": 2, "init": "invalid"}, "init must be"),
    ],
)
def test_k_shape_invalid_params(params, match):
    """Test KShape raises informative errors for invalid parameters."""
    data = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    with pytest.raises(ValueError, match=match):
        KShape(**params).fit(data)


def test_k_shape_invalid_init_array_shape():
    """Test KShape rejects an init array with the wrong shape."""
    data = make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    bad_init = np.zeros((2, 1, 3))
    with pytest.raises(ValueError, match="If init is an array"):
        KShape(n_clusters=2, init=bad_init).fit(data)


def test_k_shape_empty_cluster_reassignment():
    """Test KShape handles clusters that become empty during iteration.

    With many clusters relative to the number of cases, the random initial
    assignment leaves some clusters with no members, exercising the
    reassignment branch in the centre update.
    """
    data = make_example_3d_numpy(6, 1, 8, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=5, random_state=4, max_iter=10)
    clusterer.fit(data)
    assert clusterer.cluster_centres_.shape == (5, 1, 8)
    assert clusterer.labels_.shape == (6,)


def test_k_shape_get_test_params():
    """Test the testing parameter set is valid."""
    params = KShape._get_test_params()
    assert params["n_clusters"] == 2
    assert KShape(**params).fit(
        make_example_3d_numpy(10, 1, 8, return_y=False, random_state=1)
    )


@pytest.mark.skipif(not MULTITHREAD_TESTING, reason="Only run on multithread testing")
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_k_shape_threaded(n_jobs):
    """Test KShape threaded functionality."""
    data = make_example_3d_numpy(20, 1, 10, return_y=False, random_state=1)
    clusterer = KShape(n_clusters=2, random_state=1, max_iter=5, n_jobs=n_jobs)
    preds = clusterer.fit_predict(data)
    assert preds.shape == (20,)
    assert clusterer.cluster_centres_.shape == (2, 1, 10)


def test_shift_zeropad_1d():
    """Test the zero-padding shift helper."""
    x = np.arange(5.0)
    # no shift returns the same values
    assert np.array_equal(_shift_zeropad_1d(x, 0), x)
    # positive shift moves values right and zero-pads the left
    assert np.array_equal(_shift_zeropad_1d(x, 2), [0, 0, 0, 1, 2])
    # negative shift moves values left and zero-pads the right
    assert np.array_equal(_shift_zeropad_1d(x, -2), [2, 3, 4, 0, 0])
    # a shift larger than the series gives all zeros
    assert np.array_equal(_shift_zeropad_1d(x, 10), np.zeros(5))
    assert np.array_equal(_shift_zeropad_1d(x, -10), np.zeros(5))


def test_ncc_time_major_zero_norm():
    """Test normalised cross-correlation returns zeros for a zero series."""
    x = np.zeros((4, 1))
    y = np.ones((4, 1))
    result = _ncc_time_major(x, y)
    assert result.shape == (2 * 4 - 1,)
    assert np.all(result == 0.0)


def test_sbd_align_1d_zero_centre():
    """Test alignment against a zero centre returns the input unchanged."""
    x = np.arange(5.0)
    aligned = _sbd_align_1d(np.zeros(5), x)
    assert np.array_equal(aligned, x)


def test_shape_extraction_invalid():
    """Test shape extraction validates its input array."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _shape_extraction(np.zeros((0, 5)), np.eye(5))
    with pytest.raises(ValueError, match="must be 2D"):
        _shape_extraction(np.zeros(5), np.eye(5))
