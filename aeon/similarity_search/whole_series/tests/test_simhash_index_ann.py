"""Tests for SimHashIndexANN (multi-table LSH)."""

import warnings

import numpy as np
import pytest

from aeon.distances import euclidean_distance, pairwise_distance
from aeon.similarity_search.whole_series._commons import _bucket_dicts
from aeon.similarity_search.whole_series._simhash_index_ann import (
    SimHashIndexANN,
    _collection_to_signature,
    _series_to_signature,
    _signatures_to_keys,
)
from aeon.similarity_search.whole_series.tests.test_commons import emptied as _emptied
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.numba.general import z_normalise_series_2d, z_normalise_series_3d

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
    rng = np.random.default_rng(0)
    X = make_example_3d_numpy(n_cases=5, n_channels=2, n_timepoints=10, return_y=False)
    hash_funcs = _flatten(rng.standard_normal((8, 2, 10)))
    res = _collection_to_signature(X, hash_funcs)
    assert res.shape == (5, 8)
    assert res.dtype == np.bool_


def test_collection_matches_per_series_signature():
    """Collection hashing agrees with hashing each series individually."""
    rng = np.random.default_rng(1)
    X = make_example_3d_numpy(n_cases=6, n_channels=2, n_timepoints=10, return_y=False)
    hash_funcs = _flatten(rng.standard_normal((8, 2, 10)))
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
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=6, n_bits_per_table=8, random_state=0)
    rp.fit(X)

    assert hasattr(rp, "tables_")
    assert len(_bucket_dicts(rp.tables_)) == 6
    assert rp.tables_.case_indices.shape == (6, 20)
    assert hasattr(rp, "hash_funcs_")
    assert rp.n_cases_ == 20
    assert rp.n_channels_ == 2
    assert rp.n_timepoints_ == 50


def test_fit_hash_funcs_shape():
    """Hash functions span the full series and number n_tables * n_bits_per_table."""
    X = make_example_3d_numpy(n_cases=10, n_channels=3, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=8, n_bits_per_table=5, random_state=0)
    rp.fit(X)

    assert rp.hash_funcs_.shape == (40, 3, 50)


def test_fit_gaussian_is_default():
    """Default distribution is gaussian (real-valued, unbounded vectors)."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=8, random_state=0)
    rp.fit(X)

    assert rp.hash_func_distribution == "gaussian"
    assert rp.hash_funcs_.max() > 1.0
    assert rp.hash_funcs_.min() < -1.0


def test_fit_discrete_distribution():
    """Discrete vectors contain only -1 and 1."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
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
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(
        n_tables=4, n_bits_per_table=8, hash_func_distribution="uniform", random_state=0
    )
    rp.fit(X)

    assert rp.hash_funcs_.min() >= -1.0
    assert rp.hash_funcs_.max() <= 1.0
    assert not np.all(np.isin(rp.hash_funcs_, [-1.0, 1.0]))


def test_fit_invalid_distribution_raises():
    """An unknown hash_func_distribution raises a clear ValueError."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(hash_func_distribution="not_a_distribution", random_state=0)
    with pytest.raises(ValueError, match="hash_func_distribution must be one of"):
        rp.fit(X)


def test_fit_too_many_bits_raises():
    """n_bits_per_table above the 64-bit key width raises a clear ValueError."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_bits_per_table=65, random_state=0)
    with pytest.raises(ValueError, match="n_bits_per_table must be between 1 and 64"):
        rp.fit(X)


@pytest.mark.parametrize("n_tables", [0, -2])
def test_fit_invalid_n_tables_raises(n_tables):
    """n_tables below 1 raises a ValueError whose message names n_tables."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=n_tables, random_state=0)
    with pytest.raises(ValueError, match="n_tables"):
        rp.fit(X)


@pytest.mark.parametrize("n_tables", [2.0, 4.5, True, "4"])
def test_fit_non_integer_n_tables_raises(n_tables):
    """A non-integer n_tables (float, bool, str) raises a TypeError naming it."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=n_tables, random_state=0)
    with pytest.raises(TypeError, match="n_tables"):
        rp.fit(X)


@pytest.mark.parametrize("n_bits_per_table", [8.0, 4.5, True, "8"])
def test_fit_non_integer_n_bits_per_table_raises(n_bits_per_table):
    """A non-integer n_bits_per_table (float, bool, str) raises a TypeError."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_bits_per_table=n_bits_per_table, random_state=0)
    with pytest.raises(TypeError, match="n_bits_per_table"):
        rp.fit(X)


def test_fit_accepts_numpy_integer_params():
    """NumPy integer scalars are valid n_tables and n_bits_per_table values."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(
        n_tables=np.int64(4), n_bits_per_table=np.int32(4), random_state=0
    )
    rp.fit(X)
    assert len(_bucket_dicts(rp.tables_)) == 4


def test_fit_reproducibility():
    """Same random_state produces the same hash functions."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    r1 = SimHashIndexANN(n_tables=4, n_bits_per_table=8, random_state=42).fit(X)
    r2 = SimHashIndexANN(n_tables=4, n_bits_per_table=8, random_state=42).fit(X)

    np.testing.assert_array_equal(r1.hash_funcs_, r2.hash_funcs_)


def test_fit_all_series_indexed_in_each_table():
    """Every series appears in exactly one bucket of every table."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=6, random_state=0).fit(X)

    for table in _bucket_dicts(rp.tables_):
        indexed = set()
        total = 0
        for bucket in table.values():
            indexed.update(bucket)
            total += len(bucket)
        assert indexed == set(range(20))
        assert total == 20  # partition: each case in exactly one bucket


def test_build_index_buckets_match_dict_loop_reference():
    """The sort-based bucket build equals the original per-case dict-loop construction.

    The buckets must be array-equal (same keys, same index order within each bucket)
    to the ``setdefault(...).append(case_idx)`` construction, across several fitted
    collections and both normalize settings.
    """
    for seed in range(6):
        X = make_example_3d_numpy(
            n_cases=40,
            n_channels=2,
            n_timepoints=30,
            return_y=False,
        )
        for normalize in (True, False):
            rp = SimHashIndexANN(
                n_tables=10,
                n_bits_per_table=5,
                random_state=seed,
                normalize=normalize,
            ).fit(X)

            Xn = z_normalise_series_3d(X) if normalize else X
            signatures = _collection_to_signature(Xn, rp.hash_funcs_flat_)
            keys = _signatures_to_keys(signatures, rp.n_tables, rp.n_bits_per_table)
            for t in range(rp.n_tables):
                reference = {}
                for case_idx, key in enumerate(keys[:, t].tolist()):
                    reference.setdefault(key, []).append(case_idx)
                got = _bucket_dicts(rp.tables_)[t]
                assert set(got.keys()) == set(reference.keys())
                for key in reference:
                    np.testing.assert_array_equal(
                        np.asarray(got[key]), np.asarray(reference[key])
                    )


# =============================================================================
# Tests for predict
# =============================================================================


def test_predict_returns_correct_shape():
    """Predict returns equal-length index and distance arrays."""
    X = make_example_3d_numpy(n_cases=50, n_channels=2, n_timepoints=60, return_y=False)
    rp = SimHashIndexANN(n_tables=8, n_bits_per_table=4, random_state=0).fit(X)

    idx, dist = rp.predict(X[0], k=5)
    assert len(idx) == len(dist)
    assert 1 <= len(idx) <= 5


def test_predict_distances_are_inverse_collision_count():
    """Distances are 1 / (number of tables the neighbor collided in)."""
    X = make_example_3d_numpy(n_cases=50, n_channels=2, n_timepoints=60, return_y=False)
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
    X = make_example_3d_numpy(n_cases=40, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(n_tables=10, n_bits_per_table=4, random_state=0).fit(X)

    _, dist = rp.predict(X[0], k=5)
    assert np.all(np.diff(dist) >= 0)


@pytest.mark.parametrize(
    "n_cases,n_tables,n_bits",
    [
        # loose buckets: many hits per query -> dense bincount tally branch
        (40, 10, 5),
        # selective buckets over more cases: few hits -> sparse unique tally branch
        (200, 10, 16),
    ],
)
def test_predict_ranking_matches_dense_bincount_reference(n_cases, n_tables, n_bits):
    """Both tally branches of the hybrid ranking equal the dense bincount reference.

    ``_gather_candidates`` tallies collisions with ``np.bincount`` when the probed
    buckets yield many hits and ``np.unique`` when they yield few; the two
    parametrizations pin one regime each. The returned candidate order (collision
    count descending, index ascending tie-break) and proxy distances must match the
    dense ``bincount``/``nonzero``/``lexsort`` ranking, across several collections,
    queries and k values, for both normalize settings.
    """

    def _reference_rank(rp, query, k):
        q = z_normalise_series_2d(query) if rp.normalize else query
        signature = _series_to_signature(q, rp.hash_funcs_flat_)
        keys = _signatures_to_keys(
            signature[None, :], rp.n_tables, rp.n_bits_per_table
        )[0]
        buckets = _bucket_dicts(rp.tables_)
        hit_arrays = []
        for t in range(rp.n_tables):
            bucket = buckets[t].get(int(keys[t]))
            if bucket is not None:
                hit_arrays.append(bucket)
        if not hit_arrays:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        counts = np.bincount(np.concatenate(hit_arrays), minlength=rp.n_cases_)
        cand = np.nonzero(counts)[0]
        coll = counts[cand]
        order = np.lexsort((cand, -coll))
        n_found = min(k, len(cand))
        order = order[:n_found]
        return cand[order], 1.0 / coll[order]

    for seed in range(6):
        X = make_example_3d_numpy(
            n_cases=n_cases,
            n_channels=2,
            n_timepoints=30,
            return_y=False,
        )
        for normalize in (True, False):
            rp = SimHashIndexANN(
                n_tables=n_tables,
                n_bits_per_table=n_bits,
                random_state=seed,
                normalize=normalize,
            ).fit(X)
            for qi in (0, 5, 20):
                for k in (1, 3, 5):
                    got_idx, got_dist = rp.predict(X[qi], k=k)
                    exp_idx, exp_dist = _reference_rank(rp, X[qi], k)
                    np.testing.assert_array_equal(got_idx, exp_idx)
                    np.testing.assert_allclose(got_dist, exp_dist)


def test_predict_self_match():
    """Query collides with itself in every table, so it ranks first."""
    X = make_example_3d_numpy(n_cases=50, n_channels=2, n_timepoints=60, return_y=False)
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
    X = make_example_3d_numpy(n_cases=30, n_channels=1, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=8, n_bits_per_table=4, random_state=0).fit(X)

    idx, dist = rp.predict(X[0, 0, :], k=3)
    assert len(idx) == len(dist)


def test_predict_inverse_distance_raises():
    """inverse_distance is not supported by a near-neighbor bucket index."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(random_state=0).fit(X)
    with pytest.raises(NotImplementedError):
        rp.predict(X[0], k=3, inverse_distance=True)


def test_predict_k_larger_than_n_cases_warns():
    """A warning is raised when k exceeds the number of indexed cases."""
    X = make_example_3d_numpy(n_cases=5, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=3, random_state=0).fit(X)

    with pytest.warns(UserWarning, match="larger than"):
        idx, _ = rp.predict(X[0], k=10)
    assert len(idx) <= 5


def test_predict_empty_candidates_warns():
    """An empty candidate set warns and returns no neighbors."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(n_tables=5, n_bits_per_table=8, random_state=0).fit(X)
    rp.tables_ = _emptied(rp.tables_)

    with pytest.warns(UserWarning):
        idx, dist = rp.predict(X[0], k=3)
    assert len(idx) == 0
    assert len(dist) == 0


# =============================================================================
# Tests for algorithmic correctness
# =============================================================================


def test_identical_series_same_bucket():
    """Identical series collide in the same bucket of every table."""
    base = make_example_3d_numpy(
        n_cases=1, n_channels=2, n_timepoints=40, return_y=False
    )
    X = np.vstack([base, base, base])
    rp = SimHashIndexANN(
        n_tables=4, n_bits_per_table=8, random_state=0, normalize=False
    ).fit(X)

    for table in _bucket_dicts(rp.tables_):
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_hash_funcs_flat_dtype_follows_input(dtype):
    """The hashing matrix adopts the fitted data's floating precision."""
    X = make_example_3d_numpy(
        n_cases=20, n_channels=2, n_timepoints=50, return_y=False
    ).astype(dtype)
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
# Tests for edge cases
# =============================================================================


def test_single_series():
    """Index works with a single series."""
    X = make_example_3d_numpy(n_cases=1, n_channels=2, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=4, random_state=0).fit(X)

    with pytest.warns(UserWarning):
        idx, _ = rp.predict(X[0], k=5)
    assert len(idx) == 1
    assert idx[0] == 0


def test_high_dimensional():
    """Index works with many channels."""
    X = make_example_3d_numpy(
        n_cases=20, n_channels=10, n_timepoints=50, return_y=False
    )
    rp = SimHashIndexANN(n_tables=6, n_bits_per_table=4, random_state=0).fit(X)

    assert rp.hash_funcs_.shape[1] == 10
    idx, dist = rp.predict(X[0], k=5)
    assert len(idx) == len(dist)


def test_predict_wrong_query_length_raises():
    """A query whose length differs from the fitted series must raise."""
    X = make_example_3d_numpy(n_cases=8, n_channels=1, n_timepoints=50, return_y=False)
    rp = SimHashIndexANN(random_state=0).fit(X)
    bad_query = make_example_3d_numpy(
        n_cases=1, n_channels=1, n_timepoints=30, return_y=False
    )[0]
    with pytest.raises(ValueError, match="timepoints"):
        rp.predict(bad_query)


# =============================================================================
# Tests for the optional re-ranking distance
# =============================================================================


def test_rerank_distance_defaults_to_none():
    """The re-ranking distance is opt-in: unset means collision-count ranking."""
    assert SimHashIndexANN().rerank_distance is None


def test_predict_rerank_none_keeps_the_collision_count_ranking():
    """Leaving rerank_distance unset changes nothing about predict.

    An instance built without the parameter and one built with an explicit
    ``rerank_distance=None`` must agree, and both must still return the proxy
    distances ``1 / collision_count`` rather than true distances.
    """
    X = make_example_3d_numpy(n_cases=40, n_channels=2, n_timepoints=40, return_y=False)
    common = dict(n_tables=10, n_bits_per_table=4, random_state=0)
    default = SimHashIndexANN(**common).fit(X)
    explicit = SimHashIndexANN(**common, rerank_distance=None).fit(X)

    for query_index in (0, 5, 20):
        idx_default, dist_default = default.predict(X[query_index], k=5)
        idx_explicit, dist_explicit = explicit.predict(X[query_index], k=5)
        np.testing.assert_array_equal(idx_default, idx_explicit)
        np.testing.assert_allclose(dist_default, dist_explicit)
        # still the reciprocal of an integer collision count in [1, n_tables]
        inv = 1.0 / dist_default
        np.testing.assert_allclose(inv, np.round(inv))
        assert np.all(inv >= 1) and np.all(inv <= 10)


def test_predict_rerank_string_distance_returns_true_distances():
    """A string rerank_distance returns real distances to the query, ascending."""
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(
        n_tables=10, n_bits_per_table=4, rerank_distance="dtw", random_state=0
    ).fit(X)

    idx, dist = rp.predict(X[0], k=5)
    # Reference built from an independently normalized collection, not from rp.X_:
    # comparing against the estimator's own stored array would pass whatever
    # scaling it happened to keep.
    expected = pairwise_distance(
        z_normalise_series_3d(X)[idx],
        z_normalise_series_2d(X[0])[np.newaxis],
        method="dtw",
    ).reshape(-1)
    np.testing.assert_allclose(dist, expected)
    assert np.all(np.diff(dist) >= 0)


def test_predict_rerank_callable_distance():
    """A callable rerank_distance is accepted at fit and used by predict."""
    X = make_example_3d_numpy(n_cases=30, n_channels=1, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(
        n_tables=10,
        n_bits_per_table=4,
        rerank_distance=euclidean_distance,
        random_state=0,
    ).fit(X)

    idx, dist = rp.predict(X[0], k=3)
    expected = pairwise_distance(
        z_normalise_series_3d(X)[idx],
        z_normalise_series_2d(X[0])[np.newaxis],
        method=euclidean_distance,
    ).reshape(-1)
    np.testing.assert_allclose(dist, expected)


def test_predict_rerank_self_match_is_at_distance_zero():
    """Re-ranking scores the self-match of a query taken from the collection at 0."""
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(
        n_tables=10, n_bits_per_table=4, rerank_distance="euclidean", random_state=0
    ).fit(X)

    idx, dist = rp.predict(X[3], k=1)
    assert idx[0] == 3
    np.testing.assert_allclose(dist[0], 0.0, atol=1e-8)


def test_fit_unknown_rerank_distance_raises_before_building_the_index():
    """A typo'd rerank_distance fails at fit, not at the first predict.

    Building the index is the expensive half of this estimator, so the failure
    must come before any table is built.
    """
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=4, rerank_distance="dwt")
    with pytest.raises(ValueError, match="Invalid rerank_distance 'dwt'"):
        rp.fit(X)
    assert not hasattr(rp, "tables_")


def test_fit_stores_the_collection_on_the_searched_scale():
    """``X_`` holds the collection on the scale queries are compared on.

    ``normalize`` alone decides that scale, never ``rerank_distance``: re-ranking
    scores candidates by indexing straight into ``X_``, so if storage depended on
    ``rerank_distance`` then enabling it after fit would score a normalized query
    against a raw collection.
    """
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=40, return_y=False)
    X = 5.0 * X + 10.0  # far from zero mean, so normalizing is not a near no-op
    common = dict(n_tables=4, n_bits_per_table=4, random_state=0)
    normalized = z_normalise_series_3d(X)

    for kwargs in (
        dict(rerank_distance="euclidean", normalize=True),
        dict(normalize=True),
    ):
        est = SimHashIndexANN(**common, **kwargs).fit(X)
        np.testing.assert_allclose(
            est.X_, normalized, err_msg=f"X_ not normalized for {kwargs}"
        )

    for kwargs in (
        dict(rerank_distance="euclidean", normalize=False),
        dict(normalize=False),
    ):
        est = SimHashIndexANN(**common, **kwargs).fit(X)
        np.testing.assert_allclose(est.X_, X, err_msg=f"X_ altered by {kwargs}")


def test_fit_failure_stores_nothing():
    """A parameter error raises before ``fit`` attaches the collection.

    Validation runs from ``_validate_fit_params``, i.e. before the base ``fit``
    stores ``X_``, so a rejected estimator does not keep the whole collection
    referenced.
    """
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=40, return_y=False)
    for kwargs, match in (
        (dict(rerank_distance="dwt"), "Invalid rerank_distance 'dwt'"),
        (dict(n_tables=0), "n_tables must be a positive integer"),
        (dict(n_bits_per_table=65), "n_bits_per_table must be between 1 and 64"),
        (dict(hash_func_distribution="nope"), "hash_func_distribution must be one of"),
    ):
        est = SimHashIndexANN(**kwargs)
        with pytest.raises((ValueError, TypeError), match=match):
            est.fit(X)
        assert not hasattr(est, "X_"), f"X_ left behind by {kwargs}"
        assert not hasattr(est, "tables_")


def test_predict_normalize_is_frozen_at_fit():
    """Setting ``normalize`` on a fitted estimator does not change predictions.

    The scale of ``X_`` is fixed at fit, so honouring a later change of the flag
    would z-normalize the query against a collection that is no longer on that
    scale. The flag is read once, into ``_normalize``, and predictions are
    unaffected until the estimator is refitted.
    """
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    X = 5.0 * X + 10.0
    est = SimHashIndexANN(
        n_tables=10,
        n_bits_per_table=4,
        rerank_distance="euclidean",
        random_state=0,
        normalize=True,
    ).fit(X)

    before_idx, before_dist = est.predict(X[3], k=3)
    est.set_params(normalize=False)
    after_idx, after_dist = est.predict(X[3], k=3)

    np.testing.assert_array_equal(before_idx, after_idx)
    np.testing.assert_allclose(before_dist, after_dist)
    # The self-match stays at distance zero rather than picking up the scale of
    # the un-normalized query.
    assert after_idx[0] == 3
    np.testing.assert_allclose(after_dist[0], 0.0, atol=1e-8)


def test_predict_k_inf_returns_every_candidate_without_warning():
    """``k=np.inf`` is the documented "return all matches" sentinel, not a mistake.

    It must not trigger the "k is larger than the number of indexed cases"
    warning, which is meant for an oversized integer ``k``.
    """
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    est = SimHashIndexANN(n_tables=10, n_bits_per_table=4, random_state=0).fit(X)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idx, dist = est.predict(X[0], k=np.inf)
    assert not any(
        "larger than the number of indexed cases" in str(w.message) for w in caught
    )
    assert len(idx) == len(dist)
    assert len(idx) <= est.n_cases_


def test_predict_rerank_is_correct_when_set_after_fit():
    """Re-ranking turned on after fit still scores on the query's scale.

    A regression test for making the scale of the scored collection depend on
    ``rerank_distance`` at fit time: an estimator fitted without re-ranking and
    switched to it afterwards would then silently compare a normalized query
    against a raw collection, self-matching at a large distance instead of zero.
    """
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    X = 5.0 * X + 10.0
    est = SimHashIndexANN(
        n_tables=10, n_bits_per_table=4, random_state=0, normalize=True
    ).fit(X)

    est.set_params(rerank_distance="euclidean")
    idx, dist = est.predict(X[3], k=3)
    assert idx[0] == 3
    np.testing.assert_allclose(dist[0], 0.0, atol=1e-8)


def test_predict_rerank_scores_query_and_collection_on_the_same_scale():
    """A re-ranked distance is computed on the normalization the query got.

    The query is z-normalized before it is scored, so the fitted collection must
    be as well. Scoring a normalized query against the raw collection is a
    different, wrong number: the raw reference is asserted to differ, so this
    test genuinely discriminates between the two.
    """
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    # Push the collection well away from zero mean and unit variance so that
    # z-normalizing it is not a near no-op.
    X = 5.0 * X + 10.0
    rp = SimHashIndexANN(
        n_tables=10,
        n_bits_per_table=4,
        rerank_distance="euclidean",
        random_state=0,
        normalize=True,
    ).fit(X)

    idx, dist = rp.predict(X[0], k=3)
    query = z_normalise_series_2d(X[0])
    normalized_reference = pairwise_distance(
        z_normalise_series_3d(X)[idx], query[np.newaxis], method="euclidean"
    ).reshape(-1)
    raw_reference = pairwise_distance(
        X[idx], query[np.newaxis], method="euclidean"
    ).reshape(-1)

    np.testing.assert_allclose(dist, normalized_reference)
    assert not np.allclose(dist, raw_reference)


@pytest.mark.parametrize(
    "n_cases,n_tables,n_bits",
    [
        # loose buckets: many hits per query -> dense bincount tally branch
        (40, 10, 5),
        # selective buckets over more cases: few hits -> sparse unique tally branch
        (200, 10, 16),
    ],
)
def test_predict_rerank_ranking_matches_dense_reference(n_cases, n_tables, n_bits):
    """The re-ranked order equals an explicit dense reference.

    The reference tallies collisions with a dense bincount, scores every
    candidate with ``pairwise_distance`` and sorts by distance ascending with an
    ascending-index tie-break. Both tally branches of ``_gather_candidates`` and
    both normalize settings are covered.
    """

    def _reference_rank(rp, query, k):
        q = z_normalise_series_2d(query) if rp.normalize else query
        signature = _series_to_signature(q, rp.hash_funcs_flat_)
        keys = _signatures_to_keys(
            signature[None, :], rp.n_tables, rp.n_bits_per_table
        )[0]
        buckets = _bucket_dicts(rp.tables_)
        hit_arrays = []
        for t in range(rp.n_tables):
            bucket = buckets[t].get(int(keys[t]))
            if bucket is not None:
                hit_arrays.append(bucket)
        if not hit_arrays:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        counts = np.bincount(np.concatenate(hit_arrays), minlength=rp.n_cases_)
        cand = np.nonzero(counts)[0]
        # ``X_`` is already on the query's scale, so the candidates are scored
        # straight out of it, exactly as ``_rerank_candidates`` does.
        dists = pairwise_distance(
            rp.X_[cand], q[np.newaxis], method=rp.rerank_distance
        ).reshape(-1)
        order = np.lexsort((cand, dists))
        return cand[order][:k], dists[order][:k]

    for seed in range(3):
        X = make_example_3d_numpy(
            n_cases=n_cases,
            n_channels=2,
            n_timepoints=30,
            return_y=False,
        )
        for normalize in (True, False):
            rp = SimHashIndexANN(
                n_tables=n_tables,
                n_bits_per_table=n_bits,
                rerank_distance="euclidean",
                random_state=seed,
                normalize=normalize,
            ).fit(X)
            for qi in (0, 5, 20):
                for k in (1, 3, 5):
                    got_idx, got_dist = rp.predict(X[qi], k=k)
                    exp_idx, exp_dist = _reference_rank(rp, X[qi], k)
                    np.testing.assert_array_equal(got_idx, exp_idx)
                    np.testing.assert_allclose(got_dist, exp_dist)


def test_predict_rerank_empty_candidates_warns():
    """An empty candidate set warns and returns nothing on the re-ranking path."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(
        n_tables=5, n_bits_per_table=8, rerank_distance="euclidean", random_state=0
    ).fit(X)
    rp.tables_ = _emptied(rp.tables_)

    with pytest.warns(UserWarning, match="No candidates"):
        idx, dist = rp.predict(X[0], k=3)
    assert len(idx) == 0
    assert len(dist) == 0


def test_predict_rerank_breaks_distance_ties_by_ascending_index():
    """Equal distances are ordered by ascending index, not by candidate order.

    The dense reference above sorts exactly as the implementation does, and random
    data never ties, so neither pins the tie-break. Duplicated series force real
    ties: ``np.argsort`` alone would be free to return them in any order.
    """
    base = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=40, return_y=False
    )
    # Every series appears four times, so each distance has a four-way tie.
    X = np.concatenate([base, base, base, base], axis=0)
    rp = SimHashIndexANN(
        n_tables=10, n_bits_per_table=2, rerank_distance="euclidean", random_state=0
    ).fit(X)

    idx, dist = rp.predict(X[7], k=4)
    # The four copies of series 7 (indices 2, 7, 12, 17) are all at distance 0.
    np.testing.assert_allclose(dist, 0.0, atol=1e-8)
    np.testing.assert_array_equal(idx, [2, 7, 12, 17])


def test_predict_rerank_too_few_candidates_warns():
    """The re-ranking path warns when fewer than k candidates collided."""
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(
        n_tables=2, n_bits_per_table=20, rerank_distance="euclidean", random_state=0
    ).fit(X)
    # 20 bits over 30 series makes every bucket a singleton, so the query only
    # ever collides with itself and k=3 cannot be satisfied.
    assert all(
        len(bucket) == 1
        for table in _bucket_dicts(rp.tables_)
        for bucket in table.values()
    )

    with pytest.warns(UserWarning, match="fewer than the requested"):
        idx, _ = rp.predict(X[0], k=3)
    assert len(idx) < 3


def test_fit_unhashable_rerank_distance_raises_a_named_error():
    """An unhashable rerank_distance names the parameter it came from.

    The name lookup raises TypeError rather than ValueError for these, which would
    otherwise surface as a bare "unhashable type" with no mention of the parameter.
    """
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=40, return_y=False)
    rp = SimHashIndexANN(n_tables=4, n_bits_per_table=4, rerank_distance=["dtw"])
    with pytest.raises(ValueError, match="Invalid rerank_distance"):
        rp.fit(X)
    assert not hasattr(rp, "tables_")


def test_get_test_params_covers_both_ranking_modes():
    """The check harness must exercise the collision and the re-ranking paths."""
    params = SimHashIndexANN._get_test_params()
    assert isinstance(params, list) and len(params) == 2
    assert "rerank_distance" not in params[0]
    assert params[1]["rerank_distance"] == "dtw"
