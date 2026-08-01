"""Tests for SSHIndexANN (Sketch, Shingle & Hash)."""

import warnings
from collections import Counter

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from aeon.distances import euclidean_distance, pairwise_distance
from aeon.similarity_search.whole_series import SSHIndexANN as PublicSSHIndexANN
from aeon.similarity_search.whole_series._commons import _bucket_dicts
from aeon.similarity_search.whole_series._ssh_index_ann import (
    SSHIndexANN,
    _hash_collection,
    _minhash_to_keys,
    _n_sketch_bits,
    _series_to_sketch,
    _sketch_to_minhash,
    _splitmix64,
)
from aeon.similarity_search.whole_series.tests.test_commons import (
    assert_same_buckets as _assert_same_buckets,
)
from aeon.similarity_search.whole_series.tests.test_commons import emptied as _emptied
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.numba.general import z_normalise_series_2d, z_normalise_series_3d

# =============================================================================
# Reference implementation
#
# The estimator hashes a series in a single fused numba pass, so its
# intermediates -- shingle ids, occurrence ranks, element ids -- are never
# materialized and cannot be inspected. They are the part of SSH that follows
# the paper most literally, so they are restated here as a plain Python
# transcription of the paper's definitions. The tests below pin the paper's
# worked examples on this reference, and pin the kernel to the reference over
# randomized inputs; a bug would have to be made twice, in two independently
# written implementations, to pass both.
# =============================================================================

_UINT64_MASK = (1 << 64) - 1


def _reference_splitmix64(x):
    """Compute the splitmix64 finalizer in pure Python, masked to 64 bits."""
    z = (x + 0x9E3779B97F4A7C15) & _UINT64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return z ^ (z >> 31)


def _reference_sketch(x, filter_, shift):
    """Sketch one series by an explicit inner product per filter position."""
    window_length = filter_.shape[1]
    n_bits = _n_sketch_bits(x.shape[1], window_length, shift)
    flat_filter = filter_.reshape(-1)
    return np.array(
        [
            np.dot(x[:, j * shift : j * shift + window_length].reshape(-1), flat_filter)
            >= 0
            for j in range(n_bits)
        ]
    )


def _reference_shingle_ids(bits, shingle_size):
    """Pack every length-n window of a sketch into an integer, bit b worth 2**b."""
    return [
        sum(int(bits[j + b]) << b for b in range(shingle_size))
        for j in range(len(bits) - shingle_size + 1)
    ]


def _reference_ranks(row):
    """Return the occurrence index of each value within its row."""
    seen = Counter()
    out = []
    for value in row:
        out.append(seen[value])
        seen[value] += 1
    return out


def _reference_elements(ids):
    """Expand shingle ids into the element ids of the weighted multiset."""
    return [
        _reference_splitmix64(int(i) ^ _reference_splitmix64(r))
        for i, r in zip(ids, _reference_ranks(list(ids)))
    ]


def _reference_minhash(bits, shingle_size, seeds):
    """Run the whole of steps 2 and 3 on one sketch, following the paper."""
    elements = _reference_elements(_reference_shingle_ids(bits, shingle_size))
    return np.array(
        [
            min(_reference_splitmix64((e + int(s)) & _UINT64_MASK) for e in elements)
            for s in seeds
        ],
        dtype=np.uint64,
    )


def _reference_keys(minhashes, n_tables, n_hashes_per_table):
    """Fold each table's chunk of MinHash values into one bucket key."""
    keys = []
    for t in range(n_tables):
        key = 0
        for m in range(n_hashes_per_table):
            key = _reference_splitmix64(
                key ^ int(minhashes[t * n_hashes_per_table + m])
            )
        keys.append(key)
    return np.array(keys, dtype=np.uint64)


# =============================================================================
# Callers for the kernel's buffer-passing internals
# =============================================================================


def _sketch_of(x, filter_, shift):
    """Sketch one series with the kernel, allocating its output buffer."""
    bits = np.empty(_n_sketch_bits(x.shape[1], filter_.shape[1], shift), dtype=bool)
    _series_to_sketch(np.ascontiguousarray(x), filter_, shift, bits)
    return bits


def _occurrence_table(n_shingles):
    """Allocate the open-addressing scratch the kernel counts occurrences in."""
    size = 1
    while size < 2 * n_shingles:
        size *= 2
    return (
        np.empty(size, dtype=np.uint64),
        np.empty(size, dtype=np.int64),
        np.full(size, -1, dtype=np.int64),
    )


def _minhash_of(bits, shingle_size, seeds, table=None, tag=0):
    """Run steps 2 and 3 with the kernel, allocating its scratch buffers.

    ``table`` may be an existing scratch table, to exercise the reuse across
    cases that ``_hash_collection`` relies on.
    """
    bits = np.asarray(bits, dtype=bool)
    seeds = np.asarray(seeds, dtype=np.uint64)
    if table is None:
        table = _occurrence_table(len(bits) - shingle_size + 1)
    minhashes = np.empty(seeds.shape[0], dtype=np.uint64)
    _sketch_to_minhash(bits, shingle_size, seeds, minhashes, *table, tag)
    return minhashes


def _keys_of(minhashes, n_tables, n_hashes_per_table):
    """Fold MinHash values into bucket keys with the kernel."""
    keys = np.empty(n_tables, dtype=np.uint64)
    _minhash_to_keys(np.asarray(minhashes, dtype=np.uint64), n_hashes_per_table, keys)
    return keys


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


def test_series_to_sketch_paper_example():
    """Reproduce Eq. 6 of the paper: X=(1,2,4,1), r=(0.1,-0.1), delta=2."""
    x = np.array([[1.0, 2.0, 4.0, 1.0]])
    filter_ = np.array([[0.1, -0.1]])
    np.testing.assert_array_equal(_sketch_of(x, filter_, shift=2), [False, True])


def test_series_to_sketch_zero_is_true():
    """A zero inner product takes the +1 branch of Eq. 5."""
    x = np.array([[1.0, -1.0]])
    filter_ = np.array([[1.0, 1.0]])
    np.testing.assert_array_equal(_sketch_of(x, filter_, shift=1), [True])


def test_series_to_sketch_pairs_each_channel_with_its_filter_row():
    """Channel c of a window is weighted by row c of the filter.

    Were the two flattened in opposite orders, the same inputs would give True,
    so this pins a multivariate sketch to the intended pairing.
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    filter_ = np.array([[0.0, 1.0], [-1.0, 0.0]])  # dot = 2 - 3 = -1 -> False
    np.testing.assert_array_equal(_sketch_of(x, filter_, shift=1), [False])


def test_series_to_sketch_shape_and_dtype():
    """The sketch is a length N_B boolean array."""
    rng = np.random.default_rng(0)
    x = make_example_3d_numpy(n_cases=1, n_channels=2, n_timepoints=30, return_y=False)[
        0
    ]
    res = _sketch_of(x, rng.standard_normal((2, 6)), shift=3)
    assert res.shape == (_n_sketch_bits(30, 6, 3),)
    assert res.dtype == np.bool_


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_channels", [1, 3])
def test_series_to_sketch_matches_reference(seed, n_channels):
    """The kernel's fused sketch equals an explicit inner product per position."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_channels, 60))
    filter_ = rng.standard_normal((n_channels, 9))
    for shift in (1, 2, 5):
        np.testing.assert_array_equal(
            _sketch_of(x, filter_, shift), _reference_sketch(x, filter_, shift)
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_series_to_sketch_is_independent_of_input_precision(dtype):
    """The bits do not depend on the precision the caller happens to pass.

    The sketch is a sign test, so a lower-precision input must not silently move
    a bit across zero; the kernel accumulates in float64 whatever it is given.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 40))
    filter_ = rng.standard_normal((1, 8))
    np.testing.assert_array_equal(
        _sketch_of(x.astype(dtype), filter_.astype(dtype), 4),
        _sketch_of(x, filter_, 4),
    )


# =============================================================================
# Tests for shingling (step 2 of SSH)
# =============================================================================


def test_reference_shingle_ids_paper_example():
    """Reproduce the paper's weighted set for B=(+1,+1,-1,-1,+1,+1), n=2.

    The paper reports {(+1,+1): 2, (+1,-1): 1, (-1,+1): 1, (-1,-1): 1}. With bit
    b of a shingle weighted 2**b, those four patterns are ids 3, 1, 2 and 0.
    """
    bits = [True, True, False, False, True, True]
    ids = _reference_shingle_ids(bits, shingle_size=2)
    assert ids == [3, 1, 0, 2, 3]
    assert Counter(ids) == {3: 2, 1: 1, 0: 1, 2: 1}


def test_reference_shingle_ids_distinct_patterns_distinct_ids():
    """Every distinct bit pattern of length n maps to a distinct id."""
    n = 5
    first_ids = [
        _reference_shingle_ids([(v >> b) & 1 for b in range(n)], n)[0]
        for v in range(2**n)
    ]
    assert first_ids == list(range(2**n))


def test_reference_ranks_paper_example():
    """The paper's example has one repeated shingle, which gets rank 1."""
    assert _reference_ranks([3, 1, 0, 2, 3]) == [0, 0, 0, 0, 1]


# =============================================================================
# Tests for the uint64 mixer
# =============================================================================


def test_splitmix64_matches_reference():
    """The kernel's mixer equals the masked Python reference, so it wraps.

    Every multiplication overflows 64 bits, so this pins the arithmetic as
    modular: any dtype slip that widens or promotes it (a signed accumulator, a
    float) loses the low bits and diverges from the reference immediately.
    """
    for value in [0, 1, 2, 42, 2**32, 2**63, 2**64 - 1]:
        got = _splitmix64(np.uint64(value))
        assert 0 <= got < 2**64
        assert got == _reference_splitmix64(value)


def test_splitmix64_is_injective_on_a_sample():
    """Distinct inputs give distinct outputs over a large sample."""
    x = np.arange(100_000, dtype=np.uint64)
    got = np.array([_splitmix64(v) for v in x], dtype=np.uint64)
    assert len(np.unique(got)) == x.size


# =============================================================================
# Tests for MinHash and bucket-key packing (step 3 of SSH)
# =============================================================================


def _weighted_jaccard(row_a, row_b):
    """Return sum-min over sum-max of the two rows' value counts."""
    count_a, count_b = Counter(list(row_a)), Counter(list(row_b))
    keys = set(count_a) | set(count_b)
    numerator = sum(min(count_a[k], count_b[k]) for k in keys)
    denominator = sum(max(count_a[k], count_b[k]) for k in keys)
    return numerator / denominator


def _random_sketch(rng, n_bits):
    """Draw a random sketch, as step 1 would produce."""
    return rng.random(n_bits) > 0.5


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shingle_size", [1, 3, 15])
def test_sketch_to_minhash_matches_reference(seed, shingle_size):
    """The fused pass equals shingling, ranking, expanding and MinHash-ing in turn.

    This is what pins the two loop-level shortcuts inside the kernel -- the
    rolling shingle id and the open-addressing occurrence table -- to the
    definitions they optimize.
    """
    rng = np.random.default_rng(seed)
    bits = _random_sketch(rng, 40)
    seeds = rng.integers(0, 2**63, size=9).astype(np.uint64)
    np.testing.assert_array_equal(
        _minhash_of(bits, shingle_size, seeds),
        _reference_minhash(bits, shingle_size, seeds),
    )


def test_sketch_to_minhash_repeated_shingles_match_reference():
    """A sketch whose shingles nearly all repeat exercises the occurrence ranks.

    A constant sketch gives one shingle repeated n_shingles times, so every rank
    from 0 upwards is used and the expansion is the whole of the multiset.
    """
    seeds = np.random.default_rng(0).integers(0, 2**63, size=8).astype(np.uint64)
    for bits in (np.ones(20, dtype=bool), np.zeros(20, dtype=bool)):
        np.testing.assert_array_equal(
            _minhash_of(bits, 4, seeds), _reference_minhash(bits, 4, seeds)
        )


def test_sketch_to_minhash_handles_the_top_shingle_bit():
    """A 64-bit shingle sets bit 63 without overflowing into a signed type."""
    bits = np.zeros(64, dtype=bool)
    bits[63] = True
    seeds = np.random.default_rng(1).integers(0, 2**63, size=4).astype(np.uint64)
    assert _reference_shingle_ids(bits, 64) == [2**63]
    np.testing.assert_array_equal(
        _minhash_of(bits, 64, seeds), _reference_minhash(bits, 64, seeds)
    )


def test_sketch_to_minhash_shape_and_dtype():
    """There is one MinHash value per seed."""
    rng = np.random.default_rng(0)
    res = _minhash_of(_random_sketch(rng, 30), 5, rng.integers(0, 2**63, 7))
    assert res.shape == (7,)
    assert res.dtype == np.uint64


def test_sketch_to_minhash_identical_sketches_always_collide():
    """Identical multisets have identical MinHash values under every seed."""
    rng = np.random.default_rng(1)
    bits = _random_sketch(rng, 50)
    seeds = rng.integers(0, 2**63, size=200).astype(np.uint64)
    np.testing.assert_array_equal(
        _minhash_of(bits, 6, seeds), _minhash_of(bits.copy(), 6, seeds)
    )


def test_sketch_to_minhash_is_permutation_invariant():
    """Reordering the multiset leaves the MinHash unchanged: it is a set hash.

    At ``shingle_size=1`` a shingle is a single bit, so permuting the sketch
    permutes the multiset without changing it.
    """
    rng = np.random.default_rng(2)
    bits = _random_sketch(rng, 50)
    shuffled = rng.permutation(bits)
    seeds = rng.integers(0, 2**63, size=100).astype(np.uint64)
    np.testing.assert_array_equal(
        _minhash_of(bits, 1, seeds), _minhash_of(shuffled, 1, seeds)
    )


def test_sketch_to_minhash_disjoint_sketches_never_collide():
    """Multisets sharing no shingle never produce the same MinHash."""
    seeds = np.random.default_rng(3).integers(0, 2**63, size=500).astype(np.uint64)
    a = _minhash_of(np.ones(50, dtype=bool), 4, seeds)  # only the all-ones shingle
    b = _minhash_of(np.zeros(50, dtype=bool), 4, seeds)  # only the all-zeros shingle
    assert np.mean(a == b) == 0.0


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
    # Short shingles over a short sketch, so the alphabet is small, counts
    # repeat, and the weights actually matter.
    bits_a, bits_b = _random_sketch(rng, 42), _random_sketch(rng, 42)
    observed = np.mean(_minhash_of(bits_a, 3, seeds) == _minhash_of(bits_b, 3, seeds))
    expected = _weighted_jaccard(
        _reference_shingle_ids(bits_a, 3), _reference_shingle_ids(bits_b, 3)
    )
    # 5000 independent seeds give a standard error of at most 0.008.
    assert abs(observed - expected) < 0.03


def test_minhash_weighting_is_not_set_jaccard():
    """Counts matter, so equal supports with unequal counts do not always collide."""
    seeds = np.random.default_rng(7).integers(0, 2**63, size=5000).astype(np.uint64)
    # At n=1 a shingle is one bit, so these are the multisets {0: 30, 1: 10} and
    # {0: 10, 1: 30}: the same support, mirrored counts.
    bits_a = np.array([False] * 30 + [True] * 10)
    bits_b = np.array([False] * 10 + [True] * 30)
    observed = np.mean(_minhash_of(bits_a, 1, seeds) == _minhash_of(bits_b, 1, seeds))
    # Set Jaccard would be 1.0 here; the weighted one is (10 + 10) / (30 + 30).
    assert _weighted_jaccard(bits_a, bits_b) == pytest.approx(20 / 60)
    assert abs(observed - 20 / 60) < 0.03


def test_occurrence_table_is_reused_without_leaking_between_cases():
    """A scratch table carried over from another case must read as empty.

    ``_hash_collection`` allocates the occurrence table once per parallel task
    and never clears it, relying on the case tag alone to invalidate stale
    entries. Were the tag ignored, the second case here would continue the first
    one's occurrence counts and hash differently.
    """
    rng = np.random.default_rng(11)
    seeds = rng.integers(0, 2**63, size=6).astype(np.uint64)
    first, second = _random_sketch(rng, 30), _random_sketch(rng, 30)
    table = _occurrence_table(30 - 4 + 1)

    _minhash_of(first, 4, seeds, table=table, tag=0)
    reused = _minhash_of(second, 4, seeds, table=table, tag=1)
    np.testing.assert_array_equal(reused, _reference_minhash(second, 4, seeds))
    # The same sketch twice in a row is the case a missing tag check would let
    # through unnoticed, so it is checked explicitly.
    repeated = _minhash_of(second, 4, seeds, table=table, tag=2)
    np.testing.assert_array_equal(repeated, reused)


def test_minhash_to_keys_shape_and_dtype():
    """Keys are one uint64 per table."""
    rng = np.random.default_rng(4)
    keys = _keys_of(rng.integers(0, 2**63, size=12), n_tables=4, n_hashes_per_table=3)
    assert keys.shape == (4,)
    assert keys.dtype == np.uint64


def test_minhash_to_keys_matches_reference():
    """Folding a table's chunk of MinHash values follows the reference."""
    rng = np.random.default_rng(8)
    minhashes = rng.integers(0, 2**63, size=12).astype(np.uint64)
    np.testing.assert_array_equal(
        _keys_of(minhashes, 4, 3), _reference_keys(minhashes, 4, 3)
    )


def test_minhash_to_keys_equal_minhashes_equal_keys():
    """Series with the same MinHash values land in the same buckets."""
    row = np.random.default_rng(5).integers(0, 2**63, size=8).astype(np.uint64)
    np.testing.assert_array_equal(_keys_of(row, 4, 2), _keys_of(row.copy(), 4, 2))


def test_minhash_to_keys_uses_only_its_own_table_chunk():
    """Changing a hash in table 1 leaves table 0's key untouched."""
    base = np.random.default_rng(6).integers(0, 2**63, size=4).astype(np.uint64)
    changed = base.copy()
    changed[2] = changed[2] ^ np.uint64(1)  # a hash belonging to table 1
    keys_base, keys_changed = _keys_of(base, 2, 2), _keys_of(changed, 2, 2)
    assert keys_base[0] == keys_changed[0]
    assert keys_base[1] != keys_changed[1]


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

    assert len(_bucket_dicts(est.tables_)) == 6
    assert est.tables_.case_indices.shape == (6, 20)
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
    for table in _bucket_dicts(est.tables_):
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
    for table in _bucket_dicts(est.tables_):
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
    assert len(_bucket_dicts(est.tables_)) == 5


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
def test_fit_predict_is_independent_of_input_precision(dtype):
    """Fitting on float32 builds the same index as float64 and finds itself.

    The sketch decides a sign, so a lower-precision input must not move a case
    into another bucket; the kernel accumulates in float64 whatever it is given.
    """
    X = make_example_3d_numpy(n_cases=30, n_channels=2, n_timepoints=50, return_y=False)
    reference = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X)
    est = SSHIndexANN(**_fit_kwargs(n_tables=10)).fit(X.astype(dtype))

    _assert_same_buckets(reference.tables_, est.tables_)
    idx, _ = est.predict(X[0].astype(dtype), k=3)
    assert idx[0] == 0


# =============================================================================
# Tests for the parallel hashing kernel
# =============================================================================


def _kernel_args(est):
    """Return the kernel arguments a fitted estimator hashes with."""
    return (
        est.filter_,
        est.shift,
        est.shingle_size,
        est.hash_seeds_,
        est.n_tables,
        est.n_hashes_per_table,
    )


def test_hash_collection_matches_the_per_series_reference():
    """Hashing a collection equals sketching, shingling and folding each series.

    The kernel shares scratch buffers across the cases of a parallel task and
    splits the collection into blocks, so this is what pins the whole of that
    machinery to the definitions in the reference implementation.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((23, 2, 60))
    est = SSHIndexANN(**_fit_kwargs(n_tables=5, n_hashes_per_table=2)).fit(X)
    keys = _hash_collection(X, *_kernel_args(est))

    assert keys.shape == (23, 5)
    assert keys.dtype == np.uint64
    for i, x in enumerate(X):
        bits = _reference_sketch(x, est.filter_, est.shift)
        minhashes = _reference_minhash(bits, est.shingle_size, est.hash_seeds_)
        np.testing.assert_array_equal(keys[i], _reference_keys(minhashes, 5, 2))


def test_hash_collection_is_row_independent():
    """Every case hashes to the same keys alone as in the middle of a collection.

    A query is hashed by passing it as a one-row collection, which also makes
    this the path ``predict`` uses.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((37, 1, 50))
    est = SSHIndexANN(**_fit_kwargs()).fit(X)
    keys = _hash_collection(X, *_kernel_args(est))
    for i, x in enumerate(X):
        np.testing.assert_array_equal(
            keys[i], _hash_collection(x[np.newaxis], *_kernel_args(est))[0]
        )


@pytest.mark.parametrize("n_cases", [1, 2, 17, 40])
def test_hash_collection_is_thread_count_invariant(n_cases):
    """The number of threads changes nothing but the speed.

    Cases are split into blocks whose size depends on the thread count, and each
    block reuses one occurrence table across its cases, so a bug in either would
    show up as keys that depend on ``n_jobs``.
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n_cases, 1, 50))
    est = SSHIndexANN(**_fit_kwargs()).fit(X)

    previous_threads = get_num_threads()
    try:
        set_num_threads(1)
        serial = _hash_collection(X, *_kernel_args(est))
        set_num_threads(max(2, min(4, previous_threads)))
        parallel = _hash_collection(X, *_kernel_args(est))
    finally:
        set_num_threads(previous_threads)
    np.testing.assert_array_equal(serial, parallel)


def test_fit_is_thread_count_invariant():
    """n_jobs changes neither the buckets nor the answers."""
    X = make_example_3d_numpy(n_cases=25, n_channels=2, n_timepoints=50, return_y=False)
    serial = SSHIndexANN(**_fit_kwargs()).fit(X)
    parallel = SSHIndexANN(**_fit_kwargs(n_jobs=-1)).fit(X)

    _assert_same_buckets(serial.tables_, parallel.tables_)

    for query_index in (0, 12, 24):
        query = z_normalise_series_2d(X[query_index])
        np.testing.assert_array_equal(
            serial._gather_candidates(query)[0], parallel._gather_candidates(query)[0]
        )
        # k=1 always succeeds: a query drawn from the collection collides with
        # itself in every table, so this never warns about missing candidates.
        idx_serial, dist_serial = serial.predict(X[query_index], k=1)
        idx_parallel, dist_parallel = parallel.predict(X[query_index], k=1)
        np.testing.assert_array_equal(idx_serial, idx_parallel)
        np.testing.assert_allclose(dist_serial, dist_parallel)


def test_fit_restores_the_thread_count():
    """The kernel's thread cap must not leak into the rest of the process."""
    X = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=50, return_y=False)
    before = get_num_threads()
    SSHIndexANN(**_fit_kwargs(n_jobs=1)).fit(X)
    assert get_num_threads() == before


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
        buckets = _bucket_dicts(est.tables_)
        hits = [
            buckets[t][int(keys[t])]
            for t in range(est.n_tables)
            if int(keys[t]) in buckets[t]
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
    est.tables_ = _emptied(est.tables_)
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
