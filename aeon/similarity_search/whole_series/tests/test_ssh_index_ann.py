"""Tests for SSHIndexANN (Sketch, Shingle & Hash)."""

import warnings
from collections import Counter

import numpy as np
import pytest

from aeon.distances import euclidean_distance, pairwise_distance
from aeon.similarity_search.whole_series import SSHIndexANN as PublicSSHIndexANN
from aeon.similarity_search.whole_series import _ssh_index_ann
from aeon.similarity_search.whole_series._ssh_index_ann import (
    _LIVE_SHINGLE_ARRAYS,
    SSHIndexANN,
    _collection_to_sketch,
    _elements_to_minhash,
    _hash_chunk_size,
    _minhash_to_keys,
    _n_sketch_bits,
    _occurrence_ranks,
    _shingles_to_elements,
    _sketch_to_shingle_ids,
    _splitmix64,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.numba.general import z_normalise_series_2d, z_normalise_series_3d

# =============================================================================
# Tests for the sketch (step 1 of SSH)
# =============================================================================


@pytest.mark.parametrize(
    "n_timepoints,window_length,shift,expected",
    [(20, 4, 2, 9), (100, 30, 3, 24), (128, 80, 3, 17), (10, 10, 1, 1)],
)
def test_n_sketch_bits(n_timepoints, window_length, shift, expected):
    """The sketch length is floor((m - W) / delta) + 1."""
    assert _n_sketch_bits(n_timepoints, window_length, shift) == expected


def test_collection_to_sketch_paper_example():
    """Reproduce Eq. 6 of the paper: X=(1,2,4,1), r=(0.1,-0.1), delta=2."""
    X = np.array([[[1.0, 2.0, 4.0, 1.0]]])
    filter_flat = np.array([0.1, -0.1])
    res = _collection_to_sketch(X, filter_flat, window_length=2, shift=2)
    np.testing.assert_array_equal(res, [[False, True]])


def test_collection_to_sketch_zero_is_true():
    """A zero inner product takes the +1 branch of Eq. 5."""
    X = np.array([[[1.0, -1.0]]])
    filter_flat = np.array([1.0, 1.0])
    res = _collection_to_sketch(X, filter_flat, window_length=2, shift=1)
    np.testing.assert_array_equal(res, [[True]])


def test_collection_to_sketch_is_channel_major():
    """Each window is flattened channel-major, matching the filter's C order.

    Under the time-major flattening the same inputs would give True, so this
    pins the transpose inside ``_collection_to_sketch``.
    """
    X = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # channel-major flat = [1, 2, 3, 4]
    filter_flat = np.array([0.0, 1.0, -1.0, 0.0])  # dot = 2 - 3 = -1 -> False
    res = _collection_to_sketch(X, filter_flat, window_length=2, shift=1)
    np.testing.assert_array_equal(res, [[False]])


def test_collection_to_sketch_shape_and_dtype():
    """The sketch is a (n_cases, N_B) boolean array."""
    rng = np.random.default_rng(0)
    X = make_example_3d_numpy(n_cases=5, n_channels=2, n_timepoints=30, return_y=False)
    filter_flat = rng.standard_normal(2 * 6)
    res = _collection_to_sketch(X, filter_flat, window_length=6, shift=3)
    assert res.shape == (5, _n_sketch_bits(30, 6, 3))
    assert res.dtype == np.bool_


def test_collection_to_sketch_is_row_independent():
    """Sketching a one-series collection matches that row of the full sketch.

    A query is hashed by passing it as a one-row collection, so this pins the
    path the estimator actually uses for queries.
    """
    rng = np.random.default_rng(1)
    X = make_example_3d_numpy(n_cases=6, n_channels=2, n_timepoints=40, return_y=False)
    filter_flat = rng.standard_normal(2 * 8)
    collection = _collection_to_sketch(X, filter_flat, 8, 3)
    per_series = np.vstack(
        [_collection_to_sketch(x[np.newaxis], filter_flat, 8, 3)[0] for x in X]
    )
    np.testing.assert_array_equal(collection, per_series)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_collection_to_sketch_follows_filter_dtype(dtype):
    """The matmul runs at the filter's precision, leaving the bits unchanged."""
    X = make_example_3d_numpy(
        n_cases=8, n_channels=1, n_timepoints=40, return_y=False
    ).astype(dtype)
    rng = np.random.default_rng(3)
    filter64 = rng.standard_normal(8)
    res64 = _collection_to_sketch(X.astype(np.float64), filter64, 8, 4)
    res = _collection_to_sketch(X, filter64.astype(dtype), 8, 4)
    np.testing.assert_array_equal(res64, res)


# =============================================================================
# Tests for shingling (step 2 of SSH)
# =============================================================================


def test_sketch_to_shingle_ids_paper_example():
    """Reproduce the paper's weighted set for B=(+1,+1,-1,-1,+1,+1), n=2.

    The paper reports {(+1,+1): 2, (+1,-1): 1, (-1,+1): 1, (-1,-1): 1}. With bit
    b of a shingle weighted 2**b, those four patterns are ids 3, 1, 2 and 0.
    """
    bits = np.array([[True, True, False, False, True, True]])
    ids = _sketch_to_shingle_ids(bits, shingle_size=2)
    np.testing.assert_array_equal(ids, [[3, 1, 0, 2, 3]])
    assert Counter(ids[0].tolist()) == {3: 2, 1: 1, 0: 1, 2: 1}


def test_sketch_to_shingle_ids_shape_and_dtype():
    """There are N_B - n + 1 shingles, packed into uint64."""
    rng = np.random.default_rng(0)
    bits = rng.random((7, 30)) > 0.5
    ids = _sketch_to_shingle_ids(bits, shingle_size=15)
    assert ids.shape == (7, 30 - 15 + 1)
    assert ids.dtype == np.uint64


def test_sketch_to_shingle_ids_distinct_patterns_distinct_ids():
    """Every distinct bit pattern of length n maps to a distinct id."""
    n = 5
    patterns = np.array(
        [[(v >> b) & 1 for b in range(n)] for v in range(2**n)], dtype=bool
    )
    ids = _sketch_to_shingle_ids(patterns, shingle_size=n)
    np.testing.assert_array_equal(ids[:, 0], np.arange(2**n))


def test_sketch_to_shingle_ids_high_bit():
    """A 64-bit shingle sets the top bit without promoting to float."""
    bits = np.zeros((1, 64), dtype=bool)
    bits[0, 63] = True
    ids = _sketch_to_shingle_ids(bits, shingle_size=64)
    assert ids.dtype == np.uint64
    assert ids[0, 0] == np.uint64(2**63)


def test_sketch_to_shingle_ids_single_series():
    """Shingling works on a 1D sketch, as produced for a query."""
    bits = np.array([True, True, False, False, True, True])
    ids = _sketch_to_shingle_ids(bits, shingle_size=2)
    np.testing.assert_array_equal(ids, [3, 1, 0, 2, 3])


# =============================================================================
# Tests for the uint64 mixer and the multiset expansion
# =============================================================================


def _reference_splitmix64(x):
    """Compute the splitmix64 finalizer in pure Python, masked to 64 bits."""
    mask = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


def _reference_ranks(row):
    """Return the occurrence index of each value within its row."""
    seen = Counter()
    out = []
    for value in row:
        out.append(seen[value])
        seen[value] += 1
    return out


def test_splitmix64_matches_reference():
    """The numpy mixer equals the masked Python reference, so it wraps.

    Every multiplication overflows 64 bits, so this pins the arithmetic as
    modular: any dtype slip that widens or promotes it (a signed accumulator, a
    float) loses the low bits and diverges from the reference immediately.
    """
    values = [0, 1, 2, 42, 2**32, 2**63, 2**64 - 1]
    got = _splitmix64(np.array(values, dtype=np.uint64))
    expected = [_reference_splitmix64(v) for v in values]
    assert got.dtype == np.uint64
    np.testing.assert_array_equal(got, np.array(expected, dtype=np.uint64))


def test_splitmix64_is_injective_on_a_sample():
    """Distinct inputs give distinct outputs over a large sample."""
    x = np.arange(100_000, dtype=np.uint64)
    assert len(np.unique(_splitmix64(x))) == x.size


def test_occurrence_ranks_paper_example():
    """The paper's example has one repeated shingle, which gets rank 1."""
    ids = np.array([[3, 1, 0, 2, 3]], dtype=np.uint64)
    np.testing.assert_array_equal(_occurrence_ranks(ids), [[0, 0, 0, 0, 1]])


def test_occurrence_ranks_all_equal():
    """A row of one repeated value ranks 0, 1, 2 and so on."""
    ids = np.full((1, 6), 7, dtype=np.uint64)
    np.testing.assert_array_equal(_occurrence_ranks(ids), [[0, 1, 2, 3, 4, 5]])


def test_occurrence_ranks_single_position():
    """A row holding a single shingle ranks it 0."""
    ids = np.array([[9]], dtype=np.uint64)
    np.testing.assert_array_equal(_occurrence_ranks(ids), [[0]])


@pytest.mark.parametrize("seed", range(5))
def test_occurrence_ranks_matches_counter_reference(seed):
    """Vectorized ranks equal a per-row Counter walk, for every row."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 8, size=(6, 40)).astype(np.uint64)
    got = _occurrence_ranks(ids)
    for row_ids, row_got in zip(ids, got):
        np.testing.assert_array_equal(row_got, _reference_ranks(row_ids.tolist()))


def test_shingles_to_elements_is_injective_on_pairs():
    """Distinct (shingle, occurrence) pairs give distinct element ids."""
    ids, ranks = np.meshgrid(
        np.arange(64, dtype=np.uint64), np.arange(64, dtype=np.uint64), indexing="ij"
    )
    elements = _shingles_to_elements(ids, ranks)
    assert elements.dtype == np.uint64
    assert len(np.unique(elements)) == elements.size


def test_shingles_to_elements_is_position_independent():
    """The same (shingle, occurrence) pair gives the same element anywhere."""
    ids = np.array([[5, 9, 5], [9, 5, 5]], dtype=np.uint64)
    ranks = _occurrence_ranks(ids)
    elements = _shingles_to_elements(ids, ranks)
    # (5, rank 0) sits at [0, 0] and at [1, 1]
    assert elements[0, 0] == elements[1, 1]
    # (5, rank 1) sits at [0, 2] and at [1, 2]
    assert elements[0, 2] == elements[1, 2]
    assert elements[0, 0] != elements[0, 2]


# =============================================================================
# Tests for MinHash and bucket-key packing (step 3 of SSH)
# =============================================================================


def _weighted_jaccard(row_a, row_b):
    """Return sum-min over sum-max of the two rows' value counts."""
    count_a, count_b = Counter(row_a.tolist()), Counter(row_b.tolist())
    keys = set(count_a) | set(count_b)
    numerator = sum(min(count_a[k], count_b[k]) for k in keys)
    denominator = sum(max(count_a[k], count_b[k]) for k in keys)
    return numerator / denominator


def _minhash_rows(ids, seeds):
    """Run the expansion and MinHash pipeline on a (n_rows, n_pos) id array."""
    elements = _shingles_to_elements(ids, _occurrence_ranks(ids))
    return _elements_to_minhash(elements, seeds)


def test_elements_to_minhash_shape_and_dtype():
    """There is one MinHash value per seed, per row."""
    rng = np.random.default_rng(0)
    elements = rng.integers(0, 2**63, size=(4, 30)).astype(np.uint64)
    seeds = rng.integers(0, 2**63, size=7).astype(np.uint64)
    res = _elements_to_minhash(elements, seeds)
    assert res.shape == (4, 7)
    assert res.dtype == np.uint64


def test_elements_to_minhash_identical_rows_always_collide():
    """Identical multisets have identical MinHash values under every seed."""
    rng = np.random.default_rng(1)
    row = rng.integers(0, 20, size=(1, 50)).astype(np.uint64)
    seeds = rng.integers(0, 2**63, size=200).astype(np.uint64)
    res = _minhash_rows(np.vstack([row, row]), seeds)
    np.testing.assert_array_equal(res[0], res[1])


def test_elements_to_minhash_is_permutation_invariant():
    """Reordering a row leaves its MinHash unchanged: it is a multiset hash."""
    rng = np.random.default_rng(2)
    row = rng.integers(0, 20, size=50).astype(np.uint64)
    shuffled = rng.permutation(row)
    seeds = rng.integers(0, 2**63, size=100).astype(np.uint64)
    res = _minhash_rows(np.vstack([row, shuffled]), seeds)
    np.testing.assert_array_equal(res[0], res[1])


def test_elements_to_minhash_disjoint_rows_never_collide():
    """Multisets sharing no value never produce the same MinHash."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 20, size=50).astype(np.uint64)
    b = rng.integers(100, 120, size=50).astype(np.uint64)
    seeds = rng.integers(0, 2**63, size=500).astype(np.uint64)
    res = _minhash_rows(np.vstack([a, b]), seeds)
    assert np.mean(res[0] == res[1]) == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_minhash_collision_rate_matches_weighted_jaccard(seed):
    """Verify the core exactness claim: P(collision) equals sum-min / sum-max.

    Expanding a shingle of count c into (s, 0) ... (s, c - 1) makes the plain
    Jaccard of the expanded sets equal the weighted Jaccard of the counts, so
    ordinary MinHash over the expanded ids is an exact LSH for it. Without this
    test, skipping Ioffe's consistent weighted sampling would be unjustified.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**63, size=5000).astype(np.uint64)
    # A small alphabet so counts repeat and the weights actually matter.
    a = rng.integers(0, 6, size=40).astype(np.uint64)
    b = rng.integers(0, 6, size=40).astype(np.uint64)
    res = _minhash_rows(np.vstack([a, b]), seeds)
    observed = np.mean(res[0] == res[1])
    expected = _weighted_jaccard(a, b)
    # 5000 independent seeds give a standard error of at most 0.008.
    assert abs(observed - expected) < 0.03


def test_minhash_weighting_is_not_set_jaccard():
    """Counts matter, so equal supports with unequal counts do not always collide."""
    rng = np.random.default_rng(7)
    a = np.array([0] * 30 + [1] * 10, dtype=np.uint64)
    b = np.array([0] * 10 + [1] * 30, dtype=np.uint64)
    seeds = rng.integers(0, 2**63, size=5000).astype(np.uint64)
    res = _minhash_rows(np.vstack([a, b]), seeds)
    observed = np.mean(res[0] == res[1])
    # Set Jaccard would be 1.0 here; the weighted one is (10 + 10) / (30 + 30).
    assert _weighted_jaccard(a, b) == pytest.approx(20 / 60)
    assert abs(observed - 20 / 60) < 0.03


def test_minhash_to_keys_shape_and_dtype():
    """Keys are one uint64 per table."""
    rng = np.random.default_rng(4)
    minhashes = rng.integers(0, 2**63, size=(5, 12)).astype(np.uint64)
    keys = _minhash_to_keys(minhashes, n_tables=4, n_hashes_per_table=3)
    assert keys.shape == (5, 4)
    assert keys.dtype == np.uint64


def test_minhash_to_keys_equal_minhashes_equal_keys():
    """Rows with the same MinHash values land in the same buckets."""
    rng = np.random.default_rng(5)
    row = rng.integers(0, 2**63, size=(1, 8)).astype(np.uint64)
    keys = _minhash_to_keys(np.vstack([row, row]), n_tables=4, n_hashes_per_table=2)
    np.testing.assert_array_equal(keys[0], keys[1])


def test_minhash_to_keys_uses_only_its_own_table_chunk():
    """Changing a hash in table 1 leaves table 0's key untouched."""
    rng = np.random.default_rng(6)
    base = rng.integers(0, 2**63, size=(1, 4)).astype(np.uint64)
    changed = base.copy()
    changed[0, 2] = changed[0, 2] ^ np.uint64(1)  # a hash belonging to table 1
    keys = _minhash_to_keys(
        np.vstack([base, changed]), n_tables=2, n_hashes_per_table=2
    )
    assert keys[0, 0] == keys[1, 0]
    assert keys[0, 1] != keys[1, 1]


# =============================================================================
# Tests for fit
# =============================================================================


def _fit_kwargs(**overrides):
    """Return estimator parameters valid for a 50-timepoint collection.

    The sketch is 22 bits long and holds 15 shingles drawn from a 256-symbol
    alphabet, so unrelated series share few shingles and buckets stay selective.
    """
    kwargs = dict(window_length=8, shift=2, shingle_size=8, n_tables=6, random_state=0)
    kwargs.update(overrides)
    return kwargs


def test_fit_creates_index():
    """Fit builds the tables and records the pipeline sizes."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs()).fit(X)

    assert len(est.tables_) == 6
    assert all(isinstance(table, dict) for table in est.tables_)
    assert est.filter_.shape == (2, 8)
    assert est.hash_seeds_.shape == (6,)
    assert est.n_sketch_bits_ == _n_sketch_bits(50, 8, 2)
    assert est.n_shingles_ == est.n_sketch_bits_ - 8 + 1
    assert est.n_cases_ == 20
    assert est.n_channels_ == 2
    assert est.n_timepoints_ == 50


def test_fit_hash_seeds_count_follows_amplification():
    """There is one seed per (table, hash) pair."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=5, n_hashes_per_table=3)).fit(X)
    assert est.hash_seeds_.shape == (15,)


def test_fit_reproducibility():
    """The same random_state gives the same filter and the same seeds."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    a = SSHIndexANN(**_fit_kwargs(random_state=42)).fit(X)
    b = SSHIndexANN(**_fit_kwargs(random_state=42)).fit(X)
    np.testing.assert_array_equal(a.filter_, b.filter_)
    np.testing.assert_array_equal(a.hash_seeds_, b.hash_seeds_)


def test_fit_all_series_indexed_in_each_table():
    """Every case sits in exactly one bucket of every table."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs()).fit(X)
    for table in est.tables_:
        indexed = set()
        total = 0
        for bucket in table.values():
            indexed.update(bucket.tolist())
            total += len(bucket)
        assert indexed == set(range(20))
        assert total == 20


def test_fit_normalize_replaces_stored_collection():
    """Fitting with normalize=True stores the z-normalized collection."""
    X = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(normalize=True)).fit(X)
    np.testing.assert_allclose(est.X_, z_normalise_series_3d(X))
    raw = SSHIndexANN(**_fit_kwargs(normalize=False)).fit(X)
    np.testing.assert_allclose(raw.X_, X)


def test_fit_identical_series_share_every_bucket():
    """Identical series hash identically, so each table holds a single bucket."""
    base = make_example_3d_numpy(
        n_cases=1, n_channels=2, n_timepoints=50, return_y=False
    )
    X = np.vstack([base, base, base])
    est = SSHIndexANN(**_fit_kwargs(normalize=False)).fit(X)
    for table in est.tables_:
        assert len(table) == 1
        assert set(next(iter(table.values())).tolist()) == {0, 1, 2}


@pytest.mark.parametrize(
    "name",
    ["window_length", "shift", "shingle_size", "n_tables", "n_hashes_per_table"],
)
@pytest.mark.parametrize("value", [2.0, 4.5, True, "4"])
def test_fit_non_integer_params_raise(name, value):
    """A non-integer structural parameter raises a TypeError naming it."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(**{name: value}))
    with pytest.raises(TypeError, match=name):
        est.fit(X)


@pytest.mark.parametrize(
    "name",
    ["window_length", "shift", "shingle_size", "n_tables", "n_hashes_per_table"],
)
def test_fit_non_positive_params_raise(name):
    """A structural parameter below 1 raises a ValueError naming it."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(**{name: 0}))
    with pytest.raises(ValueError, match=name):
        est.fit(X)


def test_fit_accepts_numpy_integer_params():
    """NumPy integer scalars are valid structural parameters."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(window_length=np.int64(8), n_tables=np.int32(5)))
    est.fit(X)
    assert len(est.tables_) == 5


def test_fit_non_bool_normalize_raises():
    """A non-boolean normalize raises a TypeError."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(normalize="yes"))
    with pytest.raises(TypeError, match="normalize"):
        est.fit(X)


def test_fit_unknown_distance_raises_before_building_the_index():
    """A typo'd distance fails at fit, not at the first predict.

    Building the index is the expensive half of this estimator, so the failure
    must come before any table is built.
    """
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(distance="dwt"))
    with pytest.raises(ValueError, match="Invalid distance 'dwt'"):
        est.fit(X)
    assert not hasattr(est, "tables_")


def test_fit_accepts_callable_distance():
    """A callable distance passes fit-time validation and is used by predict."""
    X = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10, distance=euclidean_distance)).fit(X)
    idx, dist = est.predict(X[0], k=1)
    expected = pairwise_distance(
        est.X_[idx],
        z_normalise_series_2d(X[0])[np.newaxis],
        method=euclidean_distance,
    ).reshape(-1)
    np.testing.assert_allclose(dist, expected)


def test_fit_window_longer_than_series_raises():
    """A window_length above the series length raises a ValueError."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(window_length=60))
    with pytest.raises(ValueError, match="window_length"):
        est.fit(X)


def test_fit_shingle_size_above_64_raises():
    """A shingle_size above the 64-bit id width raises a ValueError."""
    X = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=200, return_y=False
    )
    est = SSHIndexANN(**_fit_kwargs(window_length=8, shift=1, shingle_size=65))
    with pytest.raises(ValueError, match="shingle_size must be at most 64"):
        est.fit(X)


def test_fit_sketch_shorter_than_shingle_raises():
    """A sketch too short to hold one shingle raises, quoting the sketch length."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(window_length=40, shift=5, shingle_size=10))
    with pytest.raises(ValueError, match="3 bits long"):
        est.fit(X)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_fit_predict_follows_input_precision(dtype):
    """Fitting on float32 hashes at float32 and still finds the self match."""
    X = make_example_3d_numpy(
        n_cases=30, n_channels=2, n_timepoints=50, return_y=False
    ).astype(dtype)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X)
    assert est.filter_flat_.dtype == dtype
    idx, _ = est.predict(X[0], k=3)
    assert idx[0] == 0


# =============================================================================
# Tests for the memory chunking of the hash pipeline
# =============================================================================


def test_hash_chunk_size_fills_the_budget_without_exceeding_it():
    """The chunk is the largest number of cases whose estimate fits the budget.

    This pins the per-case estimate documented in ``_hash_chunk_size``: the
    strided window block plus ``_LIVE_SHINGLE_ARRAYS`` arrays of
    ``n_shingles`` uint64 values.
    """
    # A 1024-point univariate series with W=32, delta=3 and n=15, the paper-sized
    # configuration: 331 sketch bits and 317 shingles.
    n_bits, n_shingles, n_channels, window_length, itemsize = 331, 317, 1, 32, 8
    per_case = (
        n_bits * n_channels * window_length * itemsize
        + _LIVE_SHINGLE_ARRAYS * 8 * n_shingles
    )
    chunk = _hash_chunk_size(n_bits, n_shingles, n_channels, window_length, itemsize)
    budget = _ssh_index_ann._HASH_CHUNK_BYTES
    assert chunk * per_case <= budget < (chunk + 1) * per_case


def test_hash_chunk_budget_stays_cache_sized():
    """The default budget must stay small enough to be cache-resident.

    ``_HASH_CHUNK_BYTES`` reads like a memory ceiling, so raising it looks free.
    It is not: the pipeline is memory-bound, and a chunk whose arrays spill out of
    cache is several times slower to hash. Fitting 5000 ECG5000 series took 434 ms
    at 8 MiB against 1534 ms at 64 MiB. A timing assertion would be flaky in CI,
    so the measured conclusion is pinned as a bound on the constant instead.
    """
    assert _ssh_index_ann._HASH_CHUNK_BYTES <= 16 * 1024 * 1024


def test_hash_chunk_size_shrinks_with_the_budget(monkeypatch):
    """A smaller budget yields a smaller chunk, floored at one case."""
    args = (331, 317, 1, 32, 8)
    full = _hash_chunk_size(*args)
    monkeypatch.setattr(_ssh_index_ann, "_HASH_CHUNK_BYTES", 1024 * 1024)
    smaller = _hash_chunk_size(*args)
    assert 1 <= smaller < full
    # A budget below one case cannot split a case further, so it must not give 0.
    monkeypatch.setattr(_ssh_index_ann, "_HASH_CHUNK_BYTES", 0)
    assert _hash_chunk_size(*args) == 1


def test_hash_collection_runs_in_several_chunks(monkeypatch):
    """The chunk loop really splits the collection when the budget is small.

    Counted structurally, on the number of ``_hash_chunk`` calls, rather than by
    measuring memory: a memory threshold would be flaky in CI.
    """
    X = make_example_3d_numpy(n_cases=17, n_channels=2, n_timepoints=50, return_y=False)
    original = SSHIndexANN._hash_chunk
    sizes = []

    def spy(self, chunk):
        sizes.append(chunk.shape[0])
        return original(self, chunk)

    monkeypatch.setattr(SSHIndexANN, "_hash_chunk", spy)

    SSHIndexANN(**_fit_kwargs()).fit(X)
    assert sizes == [17]  # the default budget hashes 17 short series in one pass

    sizes.clear()
    monkeypatch.setattr(_ssh_index_ann, "_HASH_CHUNK_BYTES", 1)
    SSHIndexANN(**_fit_kwargs()).fit(X)
    assert len(sizes) == 17
    assert sum(sizes) == 17


def test_hash_collection_chunking_is_transparent(monkeypatch):
    """Shrinking the memory budget changes nothing but the number of chunks.

    Every stage of the pipeline is row-independent, so a fit run one case at a
    time must build byte-identical tables and answer queries identically.
    """
    X = make_example_3d_numpy(n_cases=17, n_channels=2, n_timepoints=50, return_y=False)
    full = SSHIndexANN(**_fit_kwargs()).fit(X)
    monkeypatch.setattr(_ssh_index_ann, "_HASH_CHUNK_BYTES", 1)
    chunked = SSHIndexANN(**_fit_kwargs()).fit(X)

    assert len(chunked.tables_) == len(full.tables_)
    for table_full, table_chunked in zip(full.tables_, chunked.tables_):
        assert table_full.keys() == table_chunked.keys()
        for key, bucket in table_full.items():
            np.testing.assert_array_equal(bucket, table_chunked[key])

    for query_index in (0, 5, 16):
        query = z_normalise_series_2d(X[query_index])
        cand_full, coll_full = full._gather_candidates(query)
        cand_chunked, coll_chunked = chunked._gather_candidates(query)
        np.testing.assert_array_equal(cand_full, cand_chunked)
        np.testing.assert_array_equal(coll_full, coll_chunked)
        # k=1 always succeeds: a query drawn from the collection collides with
        # itself in every table, so this never warns about missing candidates.
        idx_full, dist_full = full.predict(X[query_index], k=1)
        idx_chunked, dist_chunked = chunked.predict(X[query_index], k=1)
        np.testing.assert_array_equal(idx_full, idx_chunked)
        np.testing.assert_allclose(dist_full, dist_chunked)


# =============================================================================
# Tests for predict
# =============================================================================


def test_predict_returns_matching_shapes():
    """Predict returns equal-length index and distance arrays."""
    X = make_example_3d_numpy(n_cases=40, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X)
    idx, dist = est.predict(X[0], k=5)
    assert len(idx) == len(dist)
    assert 1 <= len(idx) <= 5


def test_predict_returns_true_distances():
    """The returned distances are real distances to the query, ascending."""
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10, distance="dtw")).fit(X)
    idx, dist = est.predict(X[0], k=5)

    expected = pairwise_distance(
        est.X_[idx], z_normalise_series_2d(X[0])[np.newaxis], method="dtw"
    ).reshape(-1)
    np.testing.assert_allclose(dist, expected)
    assert np.all(np.diff(dist) >= 0)


def test_predict_self_match_ranks_first():
    """A query taken from the collection returns itself, at distance zero."""
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X)
    idx, dist = est.predict(X[3], k=1)
    assert idx[0] == 3
    np.testing.assert_allclose(dist[0], 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "n_cases,n_tables,n_hashes_per_table,shingle_size",
    [
        # loose buckets, many hits per query -> dense bincount tally branch
        (60, 8, 1, 4),
        # selective buckets over more cases -> sparse np.unique tally branch
        (200, 4, 3, 8),
    ],
)
def test_predict_matches_reference_ranking(
    n_cases, n_tables, n_hashes_per_table, shingle_size
):
    """Ranking equals a dense reference in both tally branches.

    The reference always tallies with a dense bincount and sorts explicitly, so
    comparing against it pins the sparse/dense hybrid in ``_gather_candidates``
    as well as the ranking.
    """

    def _reference(est, query, k):
        q = z_normalise_series_2d(query) if est.normalize else query
        keys = est._hash_series(q)
        hits = [
            est.tables_[t][int(keys[t])]
            for t in range(est.n_tables)
            if int(keys[t]) in est.tables_[t]
        ]
        if not hits:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        counts = np.bincount(np.concatenate(hits), minlength=est.n_cases_)
        cand = np.nonzero(counts)[0]
        dists = pairwise_distance(
            est.X_[cand], q[np.newaxis], method=est.distance
        ).reshape(-1)
        order = np.lexsort((cand, dists))
        order = order[: min(k, len(cand))]
        return cand[order], dists[order]

    for seed in range(4):
        X = make_example_3d_numpy(
            n_cases=n_cases, n_channels=2, n_timepoints=40, return_y=False
        )
        est = SSHIndexANN(
            window_length=6,
            shift=2,
            shingle_size=shingle_size,
            n_tables=n_tables,
            n_hashes_per_table=n_hashes_per_table,
            random_state=seed,
        ).fit(X)
        for query_index in (0, 7, 25):
            for k in (1, 3, 5):
                got_idx, got_dist = est.predict(X[query_index], k=k)
                exp_idx, exp_dist = _reference(est, X[query_index], k)
                np.testing.assert_array_equal(got_idx, exp_idx)
                np.testing.assert_allclose(got_dist, exp_dist)


def test_predict_1d_query():
    """Predict works with a 1D univariate query."""
    X = make_example_3d_numpy(n_cases=30, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X)
    idx, dist = est.predict(X[0, 0, :], k=3)
    assert len(idx) == len(dist)


def test_predict_inverse_distance_raises():
    """The inverse_distance option is rejected by a near-neighbor index."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs()).fit(X)
    with pytest.raises(NotImplementedError, match="inverse_distance"):
        est.predict(X[0], k=3, inverse_distance=True)


def test_predict_k_larger_than_n_cases_warns():
    """A k above the number of indexed cases warns and clamps."""
    X = make_example_3d_numpy(n_cases=5, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=4)).fit(X)
    with pytest.warns(UserWarning, match="larger than"):
        idx, _ = est.predict(X[0], k=10)
    assert len(idx) <= 5


def test_predict_k_inf_does_not_warn_about_exceeding_n_cases():
    """k=np.inf means 'every match', so clamping it must not warn.

    A "fewer than k candidates" warning may still fire, which is legitimate;
    only the "larger than the number of indexed cases" warning is wrong here.
    """
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idx, dist = est.predict(X[0], k=np.inf)
    assert not any("larger than" in str(w.message) for w in caught)
    assert len(idx) == len(dist)
    assert len(idx) <= 20


def test_predict_empty_candidates_warns():
    """An empty candidate set warns and returns nothing."""
    X = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs(n_tables=5)).fit(X)
    for table in est.tables_:
        table.clear()
    with pytest.warns(UserWarning, match="No candidates"):
        idx, dist = est.predict(X[0], k=3)
    assert len(idx) == 0
    assert len(dist) == 0


def test_predict_wrong_query_length_raises():
    """A query whose length differs from the fitted series must raise."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(**_fit_kwargs()).fit(X)
    bad = make_example_3d_numpy(
        n_cases=1, n_channels=1, n_timepoints=30, return_y=False
    )[0]
    with pytest.raises(ValueError, match="timepoints"):
        est.predict(bad)


def test_predict_distance_params_are_used():
    """The distance_params dict reaches the distance function."""
    X = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=50, return_y=False)
    est = SSHIndexANN(
        **_fit_kwargs(n_tables=10, distance="dtw", distance_params={"window": 0.1})
    ).fit(X)
    idx, dist = est.predict(X[0], k=3)
    expected = pairwise_distance(
        est.X_[idx],
        z_normalise_series_2d(X[0])[np.newaxis],
        method="dtw",
        window=0.1,
    ).reshape(-1)
    np.testing.assert_allclose(dist, expected)


# =============================================================================
# Tests for algorithmic correctness and the public surface
# =============================================================================


def test_shifted_series_collides_more_than_unrelated():
    """Check the paper's central claim: SSH ignores where a pattern occurs.

    A time-shifted copy of a series must share more buckets with it than an
    unrelated series does. This is the property separating SSHIndexANN from
    SimHashIndexANN, and it fails if shingling reintroduces position
    sensitivity.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 1, 300))
    shifted = np.roll(base, 30, axis=2)
    unrelated = rng.standard_normal((1, 1, 300))
    X = np.vstack([base, shifted, unrelated])

    n_tables = 50
    est = SSHIndexANN(
        window_length=16,
        shift=1,
        shingle_size=16,
        n_tables=n_tables,
        n_hashes_per_table=1,
        random_state=0,
    ).fit(X)

    candidates, collisions = est._gather_candidates(z_normalise_series_2d(X[0]))
    counts = dict(zip(candidates.tolist(), collisions.tolist()))
    assert counts[0] == n_tables  # the query collides with itself in every table
    # The shifted copy collides in 38 of 50 tables (observed, seeded), a clear
    # majority rather than a bare "more than zero": the paper's claim is near-total
    # collision, not merely a nonzero edge over the unrelated series.
    assert counts.get(1, 0) > n_tables // 2
    assert counts.get(1, 0) > counts.get(2, 0)


def test_get_test_params_is_valid_on_check_data():
    """The test parameters fit aeon's 20-timepoint estimator-check collection."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20, return_y=False)
    for params in SSHIndexANN._get_test_params():
        n_bits = _n_sketch_bits(20, params["window_length"], params["shift"])
        assert n_bits >= params["shingle_size"]
        SSHIndexANN(**params).fit(X)


def test_public_import():
    """The estimator is exported from the package namespace."""
    assert PublicSSHIndexANN is SSHIndexANN
