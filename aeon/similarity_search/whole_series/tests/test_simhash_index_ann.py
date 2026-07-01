"""Tests for SimHashIndexANN (multi-table LSH)."""

import numpy as np
import pytest

from aeon.similarity_search.whole_series._simhash_index_ann import (
    SimHashIndexANN,
    _collection_to_signature,
    _series_to_signature,
    _signatures_to_keys,
)

# =============================================================================
# Tests for the vectorized hashing functions
# =============================================================================


def _flatten(hash_funcs):
    """Flatten (n_projections, n_channels, n_timepoints) to 2D, as stored at fit."""
    return hash_funcs.reshape(hash_funcs.shape[0], -1)


def test_series_to_signature_positive():
    """A positive projection yields a True bit."""
    X = np.array([[1.0, 2.0, 3.0]])
    hash_funcs = _flatten(np.array([[[1.0, 1.0, 1.0]]]))
    np.testing.assert_array_equal(_series_to_signature(X, hash_funcs), [True])


def test_series_to_signature_negative():
    """A negative projection yields a False bit."""
    X = np.array([[1.0, 2.0, 3.0]])
    hash_funcs = _flatten(np.array([[[-1.0, -1.0, -1.0]]]))
    np.testing.assert_array_equal(_series_to_signature(X, hash_funcs), [False])


def test_series_to_signature_zero_is_true():
    """A zero projection counts as the non-negative (True) half-space."""
    X = np.array([[1.0, -1.0]])
    hash_funcs = _flatten(np.array([[[1.0, 1.0]]]))
    np.testing.assert_array_equal(_series_to_signature(X, hash_funcs), [True])


def test_series_to_signature_multivariate():
    """The projection spans all channels."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    hash_funcs = _flatten(np.array([[[1.0, 1.0], [1.0, 1.0]]]))
    np.testing.assert_array_equal(_series_to_signature(X, hash_funcs), [True])


def test_series_to_signature_deterministic():
    """Same input produces the same signature."""
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    hash_funcs = _flatten(
        np.array([[[1.0, 1.0, 1.0, 1.0]], [[-1.0, -1.0, -1.0, -1.0]]])
    )
    res1 = _series_to_signature(X, hash_funcs)
    res2 = _series_to_signature(X, hash_funcs)
    np.testing.assert_array_equal(res1, res2)


def test_series_to_signature_correctness():
    """Signature is the sign of each full-series projection."""
    X = np.array([[1.0, -1.0, 2.0, -2.0]])  # 1 channel, 4 timepoints
    hash_funcs = _flatten(
        np.array(
            [
                [[1.0, 1.0, 1.0, 1.0]],  # dot = 0 -> >= 0 -> True
                [[-1.0, 0.0, 0.0, 0.0]],  # dot = -1 -> < 0 -> False
            ]
        )
    )
    res = _series_to_signature(X, hash_funcs)
    np.testing.assert_array_equal(res, [True, False])


def test_collection_to_signature_shape():
    """Hashing a collection produces a (n_cases, n_projections) bool array."""
    X = np.random.rand(5, 2, 10)
    hash_funcs = _flatten(np.random.randn(8, 2, 10))
    res = _collection_to_signature(X, hash_funcs)
    assert res.shape == (5, 8)
    assert res.dtype == np.bool_


def test_collection_matches_per_series_signature():
    """Collection hashing agrees with hashing each series individually."""
    X = np.random.rand(6, 2, 10)
    hash_funcs = _flatten(np.random.randn(8, 2, 10))
    collection = _collection_to_signature(X, hash_funcs)
    per_series = np.vstack([_series_to_signature(x, hash_funcs) for x in X])
    np.testing.assert_array_equal(collection, per_series)


def test_signatures_to_keys_packs_bits():
    """Each table's k bits become the integer with those binary digits."""
    # 2 tables, 3 bits each. Table 0 bits 1,0,1 -> 1 + 4 = 5; table 1 bits 0,1,1 -> 6.
    sig = np.array([[True, False, True, False, True, True]])
    keys = _signatures_to_keys(sig, n_tables=2, n_bits=3)
    np.testing.assert_array_equal(keys, [[5, 6]])


def test_signatures_to_keys_distinct_chunks_distinct_keys():
    """Different bit patterns in a table map to different keys; equal ones collide."""
    sig = np.array(
        [
            [True, False, False, False],  # t0=1, t1=0
            [True, False, True, True],  # t0=1, t1=3
            [False, True, False, False],  # t0=2, t1=0
        ]
    )
    keys = _signatures_to_keys(sig, n_tables=2, n_bits=2)
    np.testing.assert_array_equal(keys[:, 0], [1, 1, 2])  # rows 0,1 share table-0 key
    np.testing.assert_array_equal(keys[:, 1], [0, 3, 0])


# =============================================================================
# Tests for fit
# =============================================================================


def test_fit_creates_index():
    """Fit creates the multi-table index."""
    X = np.random.rand(20, 2, 50)
    rp = SimHashIndexANN(n_tables=6, n_bits_per_table=8, random_state=0)
    rp.fit(X)

    assert hasattr(rp, "tables_")
    assert len(rp.tables_) == 6
    assert all(isinstance(table, dict) for table in rp.tables_)
    assert hasattr(rp, "hash_funcs_")
    assert rp.n_cases_ == 20
    assert rp.n_channels_ == 2
    assert rp.n_timepoints_ == 50


def test_fit_hash_funcs_shape():
    """Hash functions span the full series and number n_tables * n_bits_per_table."""
    X = np.random.rand(10, 3, 50)
    rp = SimHashIndexANN(n_tables=8, n_bits_per_table=5, random_state=0)
    rp.fit(X)

    assert rp.hash_funcs_.shape == (40, 3, 50)


def test_fit_gaussian_is_default():
    """Default distribution is gaussian (real-valued, unbounded vectors)."""
    X = np.random.rand(10, 2, 50)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=8, random_state=0)
    rp.fit(X)

    assert rp.hash_func_distribution == "gaussian"
    assert rp.hash_funcs_.max() > 1.0
    assert rp.hash_funcs_.min() < -1.0


def test_fit_discrete_distribution():
    """Discrete vectors contain only -1 and 1."""
    X = np.random.rand(10, 2, 50)
    rp = SimHashIndexANN(
        n_tables=4,
        n_bits_per_table=8,
        hash_func_distribution="discrete",
        random_state=0,
    )
    rp.fit(X)

    np.testing.assert_array_equal(sorted(np.unique(rp.hash_funcs_)), [-1, 1])


def test_fit_uniform_distribution():
    """Uniform vectors lie in [-1, 1] and are not restricted to {-1, 1}."""
    X = np.random.rand(10, 2, 50)
    rp = SimHashIndexANN(
        n_tables=4, n_bits_per_table=8, hash_func_distribution="uniform", random_state=0
    )
    rp.fit(X)

    assert rp.hash_funcs_.min() >= -1.0
    assert rp.hash_funcs_.max() <= 1.0
    assert not np.all(np.isin(rp.hash_funcs_, [-1.0, 1.0]))


def test_fit_invalid_distribution_raises():
    """An unknown hash_func_distribution raises a clear ValueError."""
    X = np.random.rand(10, 2, 50)
    rp = SimHashIndexANN(hash_func_distribution="not_a_distribution", random_state=0)
    with pytest.raises(ValueError, match="hash_func_distribution must be one of"):
        rp.fit(X)


def test_fit_too_many_bits_raises():
    """n_bits_per_table above the 64-bit key width raises a clear ValueError."""
    X = np.random.rand(10, 2, 50)
    rp = SimHashIndexANN(n_bits_per_table=65, random_state=0)
    with pytest.raises(ValueError, match="n_bits_per_table must be between 1 and 64"):
        rp.fit(X)


def test_fit_reproducibility():
    """Same random_state produces the same hash functions."""
    X = np.random.rand(20, 2, 50)
    r1 = SimHashIndexANN(n_tables=4, n_bits_per_table=8, random_state=42).fit(X)
    r2 = SimHashIndexANN(n_tables=4, n_bits_per_table=8, random_state=42).fit(X)

    np.testing.assert_array_equal(r1.hash_funcs_, r2.hash_funcs_)


def test_fit_all_series_indexed_in_each_table():
    """Every series appears in exactly one bucket of every table."""
    X = np.random.rand(20, 2, 50)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=6, random_state=0).fit(X)

    for table in rp.tables_:
        indexed = set()
        total = 0
        for bucket in table.values():
            indexed.update(bucket)
            total += len(bucket)
        assert indexed == set(range(20))
        assert total == 20  # partition: each case in exactly one bucket


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_hash_funcs_flat_dtype_follows_input(dtype):
    """The hashing matrix adopts the fitted data's floating precision."""
    X = np.random.rand(20, 2, 50).astype(dtype)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=6, random_state=0).fit(X)
    assert rp.hash_funcs_flat_.dtype == dtype


def test_float32_input_matches_float64_buckets():
    """Fitting in float32 recovers the same neighbors as float64 on easy data."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.01, size=(1, 1, 50)) + np.linspace(0, 1, 50)
    b = rng.normal(0.0, 0.01, size=(1, 1, 50)) - np.linspace(0, 1, 50)
    X = np.vstack([a, a + 0.001, b, b + 0.001])
    common = dict(n_tables=15, n_bits_per_table=6, random_state=0)
    idx64, _ = SimHashIndexANN(**common).fit(X.astype(np.float64)).predict(X[0], k=2)
    idx32, _ = (
        SimHashIndexANN(**common)
        .fit(X.astype(np.float32))
        .predict(X[0].astype(np.float32), k=2)
    )
    assert set(idx64) == {0, 1}
    assert set(idx32) == {0, 1}


# =============================================================================
# Tests for predict
# =============================================================================


def test_predict_returns_correct_shape():
    """Predict returns equal-length index and distance arrays."""
    X = np.random.rand(50, 2, 60)
    rp = SimHashIndexANN(n_tables=8, n_bits_per_table=4, random_state=0).fit(X)

    idx, dist = rp.predict(X[0], k=5)
    assert len(idx) == len(dist)
    assert 1 <= len(idx) <= 5


def test_predict_distances_are_inverse_collision_count():
    """Distances are 1 / (number of tables the neighbor collided in)."""
    X = np.random.rand(50, 2, 60)
    n_tables = 8
    rp = SimHashIndexANN(n_tables=n_tables, n_bits_per_table=4, random_state=0).fit(X)

    _, dist = rp.predict(X[0], k=5)
    inv = 1.0 / dist
    # every proxy distance must be the reciprocal of an integer collision count
    np.testing.assert_allclose(inv, np.round(inv))
    assert np.all(inv >= 1)
    assert np.all(inv <= n_tables)


def test_predict_distances_are_sorted():
    """Returned neighbors are ordered by increasing proxy distance."""
    X = np.random.rand(40, 2, 40)
    rp = SimHashIndexANN(n_tables=10, n_bits_per_table=4, random_state=0).fit(X)

    _, dist = rp.predict(X[0], k=5)
    assert np.all(np.diff(dist) >= 0)


def test_predict_self_match():
    """Query collides with itself in every table, so it ranks first."""
    X = np.random.rand(50, 2, 60)
    n_tables = 8
    rp = SimHashIndexANN(
        n_tables=n_tables, n_bits_per_table=4, random_state=0, normalize=False
    ).fit(X)

    idx, dist = rp.predict(X[3], k=1)
    assert idx[0] == 3
    # self collides in all n_tables tables -> proxy distance 1 / n_tables
    np.testing.assert_allclose(dist[0], 1.0 / n_tables)


def test_predict_1d_query():
    """Predict works with a 1D (univariate) query."""
    X = np.random.rand(30, 1, 50)
    rp = SimHashIndexANN(n_tables=8, n_bits_per_table=4, random_state=0).fit(X)

    idx, dist = rp.predict(X[0, 0, :], k=3)
    assert len(idx) == len(dist)


def test_predict_inverse_distance_raises():
    """inverse_distance is not supported by a near-neighbor bucket index."""
    X = np.random.rand(20, 2, 40)
    rp = SimHashIndexANN(random_state=0).fit(X)
    with pytest.raises(NotImplementedError):
        rp.predict(X[0], k=3, inverse_distance=True)


def test_predict_k_larger_than_n_cases_warns():
    """A warning is raised when k exceeds the number of indexed cases."""
    X = np.random.rand(5, 2, 40)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=3, random_state=0).fit(X)

    with pytest.warns(UserWarning, match="larger than"):
        idx, _ = rp.predict(X[0], k=10)
    assert len(idx) <= 5


def test_predict_empty_candidates_warns():
    """An empty candidate set warns and returns no neighbors."""
    X = np.random.rand(20, 2, 40)
    rp = SimHashIndexANN(n_tables=5, n_bits_per_table=8, random_state=0).fit(X)
    for table in rp.tables_:
        table.clear()

    with pytest.warns(UserWarning):
        idx, dist = rp.predict(X[0], k=3)
    assert len(idx) == 0
    assert len(dist) == 0


# =============================================================================
# Tests for algorithmic correctness
# =============================================================================


def test_identical_series_same_bucket():
    """Identical series collide in the same bucket of every table."""
    base = np.random.rand(1, 2, 40)
    X = np.vstack([base, base, base])
    rp = SimHashIndexANN(
        n_tables=4, n_bits_per_table=8, random_state=0, normalize=False
    ).fit(X)

    for table in rp.tables_:
        assert len(table) == 1
        assert set(next(iter(table.values()))) == {0, 1, 2}


def test_predict_finds_exact_nearest_on_easy_data():
    """With well-separated clusters, the index recovers the exact 1-NN."""
    rng = np.random.default_rng(0)
    # Two well-separated groups; nearest neighbor is always the same-group twin.
    a = rng.normal(0.0, 0.01, size=(1, 1, 50)) + np.linspace(0, 1, 50)
    b = rng.normal(0.0, 0.01, size=(1, 1, 50)) - np.linspace(0, 1, 50)
    X = np.vstack([a, a + 0.001, b, b + 0.001])
    rp = SimHashIndexANN(
        n_tables=15, n_bits_per_table=6, random_state=0, normalize=True
    ).fit(X)

    idx, _ = rp.predict(X[0], k=2)
    assert set(idx) == {0, 1}  # the two near-identical "a" series


# =============================================================================
# Tests for edge cases
# =============================================================================


def test_single_series():
    """Index works with a single series."""
    X = np.random.rand(1, 2, 50)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=4, random_state=0).fit(X)

    with pytest.warns(UserWarning):
        idx, _ = rp.predict(X[0], k=5)
    assert len(idx) == 1
    assert idx[0] == 0


def test_high_dimensional():
    """Index works with many channels."""
    X = np.random.rand(20, 10, 50)
    rp = SimHashIndexANN(n_tables=6, n_bits_per_table=4, random_state=0).fit(X)

    assert rp.hash_funcs_.shape[1] == 10
    idx, dist = rp.predict(X[0], k=5)
    assert len(idx) == len(dist)


def test_predict_wrong_query_length_raises():
    """A query whose length differs from the fitted series must raise."""
    X = np.random.RandomState(0).rand(8, 1, 50)
    rp = SimHashIndexANN(random_state=0).fit(X)
    bad_query = np.random.RandomState(1).rand(1, 30)
    with pytest.raises(ValueError, match="timepoints"):
        rp.predict(bad_query)
