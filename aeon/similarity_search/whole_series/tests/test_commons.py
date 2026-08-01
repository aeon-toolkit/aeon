"""Tests for the shared whole series hash index helpers."""

import numpy as np

from aeon.similarity_search.whole_series._commons import (
    _build_hash_tables,
    _tally_bucket_collisions,
)


def test_build_hash_tables_partitions_cases():
    """Each table assigns every case to exactly one bucket."""
    keys = np.array([[1, 7], [1, 8], [2, 7], [2, 7]], dtype=np.uint64)
    tables = _build_hash_tables(keys, n_tables=2)
    assert len(tables) == 2
    for table in tables:
        assert sum(len(bucket) for bucket in table.values()) == 4
        assert set(np.concatenate(list(table.values())).tolist()) == {0, 1, 2, 3}


def test_build_hash_tables_groups_equal_keys_in_case_order():
    """Cases sharing a key land in one bucket, in ascending case order."""
    keys = np.array([[5], [3], [5], [3], [5]], dtype=np.uint64)
    table = _build_hash_tables(keys, n_tables=1)[0]
    np.testing.assert_array_equal(table[5], [0, 2, 4])
    np.testing.assert_array_equal(table[3], [1, 3])


def test_build_hash_tables_bucket_dtype_is_intp():
    """Buckets are intp arrays, so np.bincount can consume them directly."""
    keys = np.array([[1], [1], [2]], dtype=np.uint64)
    table = _build_hash_tables(keys, n_tables=1)[0]
    for bucket in table.values():
        assert bucket.dtype == np.intp


def test_build_hash_tables_matches_dict_loop_reference():
    """The sort-based build equals a plain per-case dict-append loop."""
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 6, size=(40, 5)).astype(np.uint64)
    tables = _build_hash_tables(keys, n_tables=5)
    for t in range(5):
        reference = {}
        for case_idx, key in enumerate(keys[:, t].tolist()):
            reference.setdefault(key, []).append(case_idx)
        assert set(tables[t].keys()) == set(reference.keys())
        for key, expected in reference.items():
            np.testing.assert_array_equal(tables[t][key], expected)


def test_tally_bucket_collisions_counts_tables():
    """A candidate's tally is the number of tables whose bucket it shares."""
    keys = np.array([[1, 1], [1, 2], [2, 1]], dtype=np.uint64)
    tables = _build_hash_tables(keys, n_tables=2)
    # The query hits key 1 in table 0 (cases 0 and 1) and key 1 in table 1
    # (cases 0 and 2), so case 0 collides twice and cases 1 and 2 once each.
    cand, coll = _tally_bucket_collisions(tables, np.array([1, 1]), n_cases=3)
    np.testing.assert_array_equal(cand, [0, 1, 2])
    np.testing.assert_array_equal(coll, [2, 1, 1])


def test_tally_bucket_collisions_no_hits():
    """A query matching no bucket yields two empty arrays."""
    keys = np.array([[1], [1]], dtype=np.uint64)
    tables = _build_hash_tables(keys, n_tables=1)
    cand, coll = _tally_bucket_collisions(tables, np.array([99]), n_cases=2)
    assert cand.size == 0
    assert coll.size == 0


def test_tally_bucket_collisions_branches_agree():
    """The dense bincount and sparse unique branches give identical results.

    The branch is chosen by ``hits.size >= n_cases // 8``. Passing the real
    n_cases takes the dense branch; passing a much larger one takes the sparse
    branch. Both must return the same candidates and the same counts.
    """
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 4, size=(50, 6)).astype(np.uint64)
    tables = _build_hash_tables(keys, n_tables=6)
    dense = _tally_bucket_collisions(tables, keys[0], n_cases=50)
    sparse = _tally_bucket_collisions(tables, keys[0], n_cases=100_000)
    np.testing.assert_array_equal(dense[0], sparse[0])
    np.testing.assert_array_equal(dense[1], sparse[1])
