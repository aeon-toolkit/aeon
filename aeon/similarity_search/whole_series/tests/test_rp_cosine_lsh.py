"""Tests for LSHIndex."""

import numpy as np
import pytest

from aeon.similarity_search.whole_series._rp_cosine_lsh import (
    LSHIndex,
    _bool_hamming_dist,
    _bool_hamming_dist_matrix,
    _collection_to_bool,
    _dot_product_sign,
    _series_to_bool,
)

# =============================================================================
# Tests for low-level numba functions
# =============================================================================


def test_bool_hamming_dist_identical():
    """Hamming distance between identical arrays should be 0."""
    X = np.array([True, False, True, False], dtype=np.bool_)
    Y = np.array([True, False, True, False], dtype=np.bool_)
    assert _bool_hamming_dist(X, Y) == 0


def test_bool_hamming_dist_opposite():
    """Hamming distance between opposite arrays should equal length."""
    X = np.array([True, True, True, True], dtype=np.bool_)
    Y = np.array([False, False, False, False], dtype=np.bool_)
    assert _bool_hamming_dist(X, Y) == 4


def test_bool_hamming_dist_partial():
    """Hamming distance should count differing positions."""
    X = np.array([True, False, True, False], dtype=np.bool_)
    Y = np.array([True, True, False, False], dtype=np.bool_)
    # Positions 1 and 2 differ
    assert _bool_hamming_dist(X, Y) == 2


def test_bool_hamming_dist_matrix_multiple_buckets():
    """Distance matrix with multiple buckets."""
    X_bool = np.array([True, False, True, False], dtype=np.bool_)
    collection = np.array(
        [
            [True, False, True, False],  # Distance 0
            [False, False, True, False],  # Distance 1
            [False, True, False, True],  # Distance 4
        ],
        dtype=np.bool_,
    )
    result = _bool_hamming_dist_matrix(X_bool, collection)
    np.testing.assert_array_equal(result, [0, 1, 4])


def test_dot_product_sign_positive():
    """Positive dot product returns True."""
    X = np.array([[1.0, 2.0, 3.0]])
    Y = np.array([[1.0, 1.0, 1.0]])
    assert _dot_product_sign(X, Y) is True


def test_dot_product_sign_negative():
    """Negative dot product returns False."""
    X = np.array([[1.0, 2.0, 3.0]])
    Y = np.array([[-1.0, -1.0, -1.0]])
    assert _dot_product_sign(X, Y) is False


def test_dot_product_sign_zero():
    """Zero dot product returns True (>= 0)."""
    X = np.array([[1.0, -1.0]])
    Y = np.array([[1.0, 1.0]])
    assert _dot_product_sign(X, Y) is True


def test_dot_product_sign_multivariate():
    """Dot product works with multiple channels."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 channels, 2 timepoints
    Y = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert _dot_product_sign(X, Y) is True


def test_series_to_bool_deterministic():
    """Same input produces same hash."""
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    hash_funcs = np.array([[[1.0, 1.0]], [[-1.0, -1.0]]])
    start_points = np.array([0, 2])
    length = 2

    result1 = _series_to_bool(X, hash_funcs, start_points, length)
    result2 = _series_to_bool(X, hash_funcs, start_points, length)
    np.testing.assert_array_equal(result1, result2)


def test_series_to_bool_correctness():
    """Verify hash values are computed correctly."""
    X = np.array([[1.0, -1.0, 2.0, -2.0]])
    # Hash func 1: [1, 1] at start 0 -> dot([1, -1], [1, 1]) = 0 >= 0 -> True
    # Hash func 2: [-1, -1] at start 2 -> dot([2, -2], [-1, -1]) = 0 >= 0 -> True
    hash_funcs = np.array([[[1.0, 1.0]], [[-1.0, -1.0]]])
    start_points = np.array([0, 2])
    length = 2

    result = _series_to_bool(X, hash_funcs, start_points, length)
    np.testing.assert_array_equal(result, [True, True])


def test_collection_to_bool_shape():
    """Hashing multiple series produces correct shape."""
    X = np.random.rand(5, 2, 10)
    hash_funcs = np.random.randn(8, 2, 3)
    start_points = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    length = 3

    result = _collection_to_bool(X, hash_funcs, start_points, length)
    assert result.shape == (5, 8)
    assert result.dtype == np.bool_


# =============================================================================
# Tests for LSHIndex fit
# =============================================================================


def test_fit_creates_index():
    """Fit creates necessary index structures."""
    X = np.random.rand(10, 2, 50)
    lsh = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh.fit(X)

    assert hasattr(lsh, "index_")
    assert hasattr(lsh, "_raw_index_bool_arrays")
    assert hasattr(lsh, "hash_funcs_")
    assert hasattr(lsh, "start_points_")
    assert lsh.n_timepoints_ == 50
    assert lsh.n_channels_ == 2
    assert lsh.n_cases_ == 10


def test_fit_hash_funcs_shape():
    """Hash functions have correct shape."""
    X = np.random.rand(10, 3, 100)
    lsh = LSHIndex(n_hash_funcs=64, hash_func_coverage=0.25, random_state=42)
    lsh.fit(X)

    assert lsh.hash_funcs_.shape == (64, 3, 25)
    assert len(lsh.start_points_) == 64


def test_fit_discrete_vectors():
    """Discrete vectors contain only -1 and 1."""
    X = np.random.rand(10, 2, 50)
    lsh = LSHIndex(n_hash_funcs=32, use_discrete_vectors=True, random_state=42)
    lsh.fit(X)

    unique_values = np.unique(lsh.hash_funcs_)
    np.testing.assert_array_equal(sorted(unique_values), [-1, 1])


def test_fit_reproducibility():
    """Same random_state produces same index."""
    X = np.random.rand(10, 2, 50)

    lsh1 = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh1.fit(X)

    lsh2 = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh2.fit(X)

    np.testing.assert_array_equal(lsh1.hash_funcs_, lsh2.hash_funcs_)
    np.testing.assert_array_equal(lsh1.start_points_, lsh2.start_points_)


def test_fit_all_series_indexed():
    """All series are present in the index."""
    X = np.random.rand(20, 2, 50)
    lsh = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh.fit(X)

    indexed_series = set()
    for indices in lsh.index_.values():
        indexed_series.update(indices)

    assert indexed_series == set(range(20))


# =============================================================================
# Tests for LSHIndex predict
# =============================================================================


def test_predict_returns_correct_shape():
    """Predict returns arrays of correct shape."""
    X = np.random.rand(50, 2, 100)
    lsh = LSHIndex(n_hash_funcs=64, random_state=42)
    lsh.fit(X)

    idx, dist = lsh.predict(X[0], k=5)
    assert len(idx) == 5
    assert len(dist) == 5


def test_predict_self_match():
    """Query series finds itself as nearest neighbor."""
    X = np.random.rand(50, 2, 100)
    lsh = LSHIndex(n_hash_funcs=128, random_state=42, normalize=False)
    lsh.fit(X)

    idx, dist = lsh.predict(X[0], k=1)
    assert dist[0] == 0
    assert idx[0] == 0


def test_predict_k_larger_than_n_cases_warns():
    """Warning raised when k > n_cases."""
    X = np.random.rand(5, 2, 50)
    lsh = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh.fit(X)

    with pytest.warns(UserWarning, match="k=10 is larger than"):
        idx, dist = lsh.predict(X[0], k=10)

    assert len(idx) <= 5


def test_predict_1d_query():
    """Predict works with 1D query (univariate)."""
    X = np.random.rand(20, 1, 50)
    lsh = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh.fit(X)

    query = X[0, 0, :]
    idx, dist = lsh.predict(query, k=3)
    assert len(idx) == 3


def test_predict_inverse_distance():
    """inverse_distance returns dissimilar series."""
    X = np.random.rand(50, 2, 100)
    lsh = LSHIndex(n_hash_funcs=64, random_state=42)
    lsh.fit(X)

    idx_similar, _ = lsh.predict(X[0], k=5, inverse_distance=False)
    idx_dissimilar, _ = lsh.predict(X[0], k=5, inverse_distance=True)

    assert not np.array_equal(idx_similar, idx_dissimilar)


# =============================================================================
# Tests for algorithmic correctness
# =============================================================================


def test_identical_series_same_bucket():
    """Identical series should hash to the same bucket."""
    base_series = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    X = np.vstack([base_series, base_series, base_series])

    lsh = LSHIndex(n_hash_funcs=32, random_state=42, normalize=False)
    lsh.fit(X)

    assert len(lsh.index_) == 1
    bucket_contents = list(lsh.index_.values())[0]
    assert set(bucket_contents) == {0, 1, 2}


def test_opposite_series_different_buckets():
    """Series with opposite signs should have different hashes."""
    X = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
            [[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]],
        ]
    )

    lsh = LSHIndex(
        n_hash_funcs=64, hash_func_coverage=1.0, random_state=42, normalize=False
    )
    lsh.fit(X)

    idx, _ = lsh.predict(X[0], k=1)
    assert idx[0] == 0

    idx, _ = lsh.predict(X[1], k=1)
    assert idx[0] == 1


def test_similar_series_closer_hash():
    """More similar series should have smaller Hamming distance."""
    A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    B = np.array([[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]])
    C = np.array([[-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]])

    X = np.vstack([A[np.newaxis, :], B[np.newaxis, :], C[np.newaxis, :]])

    lsh = LSHIndex(
        n_hash_funcs=128, hash_func_coverage=1.0, random_state=42, normalize=False
    )
    lsh.fit(X)

    idx, _ = lsh.predict(A, k=3)
    assert idx[0] == 0
    assert list(idx).index(1) < list(idx).index(2)


def test_cosine_similarity_scale_invariance():
    """Vectors with same direction but different magnitudes hash similarly."""
    A = np.array([[1.0, 1.0, 1.0, 1.0]])
    B = np.array([[10.0, 10.0, 10.0, 10.0]])

    X = np.vstack([A[np.newaxis, :], B[np.newaxis, :]])

    lsh = LSHIndex(
        n_hash_funcs=64, hash_func_coverage=1.0, random_state=42, normalize=False
    )
    lsh.fit(X)

    assert len(lsh.index_) == 1


def test_orthogonal_vectors_hash_similarity():
    """Orthogonal vectors should have ~50% hash similarity."""
    A = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    B = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])

    X = np.vstack([A[np.newaxis, :], B[np.newaxis, :]])

    lsh = LSHIndex(
        n_hash_funcs=256, hash_func_coverage=1.0, random_state=42, normalize=False
    )
    lsh.fit(X)

    hash_A = _series_to_bool(A, lsh.hash_funcs_, lsh.start_points_, lsh.window_length_)
    hash_B = _series_to_bool(B, lsh.hash_funcs_, lsh.start_points_, lsh.window_length_)

    hamming_dist = _bool_hamming_dist(hash_A, hash_B)
    assert 64 < hamming_dist < 192


def test_normalization_effect():
    """Normalization makes scale-different series identical."""
    A = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    B = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]])

    X = np.vstack([A[np.newaxis, :], B[np.newaxis, :]])

    lsh = LSHIndex(
        n_hash_funcs=64, hash_func_coverage=1.0, random_state=42, normalize=True
    )
    lsh.fit(X)

    assert len(lsh.index_) == 1


# =============================================================================
# Tests for edge cases
# =============================================================================


def test_single_series():
    """Index works with single series."""
    X = np.random.rand(1, 2, 50)
    lsh = LSHIndex(n_hash_funcs=32, random_state=42)
    lsh.fit(X)

    with pytest.warns(UserWarning):
        idx, dist = lsh.predict(X[0], k=5)

    assert len(idx) == 1
    assert idx[0] == 0


def test_short_series():
    """Index works with very short series."""
    X = np.random.rand(10, 2, 4)
    lsh = LSHIndex(n_hash_funcs=16, hash_func_coverage=0.5, random_state=42)
    lsh.fit(X)

    assert lsh.window_length_ == 2

    idx, dist = lsh.predict(X[0], k=3)
    assert len(idx) == 3


def test_high_dimensional():
    """Index works with many channels."""
    X = np.random.rand(20, 10, 50)
    lsh = LSHIndex(n_hash_funcs=64, random_state=42)
    lsh.fit(X)

    assert lsh.hash_funcs_.shape[1] == 10

    idx, dist = lsh.predict(X[0], k=5)
    assert len(idx) == 5


def test_full_coverage():
    """hash_func_coverage=1.0 uses full series length."""
    X = np.random.rand(10, 2, 50)
    lsh = LSHIndex(n_hash_funcs=32, hash_func_coverage=1.0, random_state=42)
    lsh.fit(X)

    assert lsh.window_length_ == 50
    np.testing.assert_array_equal(lsh.start_points_, np.zeros(32))
