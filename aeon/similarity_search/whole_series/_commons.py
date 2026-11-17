"""Shared building blocks for whole series hash indexes."""

__maintainer__ = ["baraline"]

from typing import NamedTuple

import numpy as np
from numba import njit, prange


class HashTables(NamedTuple):
    """
    The hash tables of an LSH index, in a compressed sparse row layout.

    A dict per table mapping a bucket key to its array of case indices is the
    obvious representation, and it is what this replaced. It does not scale:
    with selective buckets there is nearly one bucket per case per table, so
    building the index means creating ``n_cases * n_tables`` Python objects --
    a dict entry and a small array each -- which dominated fit time and cost
    far more memory than the indices themselves. Here every table's buckets
    live in four flat arrays instead, so building the index allocates a fixed
    number of arrays whatever the collection size.

    Attributes
    ----------
    table_offsets : np.ndarray of shape (n_tables + 1,), dtype int64
        Where each table's buckets start and stop in ``bucket_keys`` and
        ``bucket_starts``.
    bucket_keys : np.ndarray of shape (n_buckets,), dtype uint64
        The distinct bucket keys, **ascending within each table**, so a query
        can find its bucket by binary search.
    bucket_starts : np.ndarray of shape (n_buckets,), dtype int64
        Where each bucket starts in its table's row of ``case_indices``. A
        bucket ends where the next one in the same table starts, and the last
        bucket of a table ends at ``n_cases``.
    case_indices : np.ndarray of shape (n_tables, n_cases), dtype intp
        Every case index, grouped by bucket and ascending within a bucket.
    """

    table_offsets: np.ndarray
    bucket_keys: np.ndarray
    bucket_starts: np.ndarray
    case_indices: np.ndarray


@njit(cache=True, parallel=True)
def _group_cases_by_key(keys):
    """
    Order the cases of each table by bucket key and count the distinct keys.

    Parameters
    ----------
    keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
        Integer bucket key of every case in every table.

    Returns
    -------
    case_indices : np.ndarray of shape (n_tables, n_cases), dtype intp
        Case indices ordered by bucket key. The sort is stable, so cases stay
        in ascending order within a bucket.
    n_buckets : np.ndarray of shape (n_tables,), dtype int64
        Number of distinct keys in each table.
    """
    n_cases, n_tables = keys.shape
    case_indices = np.empty((n_tables, n_cases), dtype=np.intp)
    n_buckets = np.empty(n_tables, dtype=np.int64)
    for t in prange(n_tables):
        # A contiguous copy of the column: ``keys`` is row-major, so a table's
        # keys are strided, and both the sort and the scan below read them
        # repeatedly.
        column = np.empty(n_cases, dtype=np.uint64)
        for i in range(n_cases):
            column[i] = keys[i, t]
        order = np.argsort(column, kind="mergesort")

        count = 0
        for r in range(n_cases):
            case_indices[t, r] = order[r]
            if r == 0 or column[order[r]] != column[order[r - 1]]:
                count += 1
        n_buckets[t] = count
    return case_indices, n_buckets


@njit(cache=True, parallel=True)
def _fill_buckets(keys, case_indices, table_offsets, bucket_keys, bucket_starts):
    """
    Write the key and the start offset of every bucket into the flat arrays.

    Parameters
    ----------
    keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
        Integer bucket key of every case in every table.
    case_indices : np.ndarray of shape (n_tables, n_cases), dtype intp
        Case indices ordered by bucket key, from ``_group_cases_by_key``.
    table_offsets : np.ndarray of shape (n_tables + 1,), dtype int64
        Where each table's buckets start, from the counts of the same call.
    bucket_keys : np.ndarray of shape (n_buckets,), dtype uint64
        Output buffer for the distinct keys.
    bucket_starts : np.ndarray of shape (n_buckets,), dtype int64
        Output buffer for the offsets into ``case_indices``.
    """
    n_cases, n_tables = keys.shape
    for t in prange(n_tables):
        position = table_offsets[t]
        for r in range(n_cases):
            key = keys[case_indices[t, r], t]
            # The r == 0 test also stops the comparison from reading the
            # previous table's last key.
            if r == 0 or key != bucket_keys[position - 1]:
                bucket_keys[position] = key
                bucket_starts[position] = r
                position += 1


def _build_hash_tables(keys, n_tables):
    """
    Group case indices into per-table buckets from their integer bucket keys.

    Parameters
    ----------
    keys : np.ndarray of shape (n_cases, n_tables)
        Integer bucket key of every case in every table.
    n_tables : int
        Number of hash tables.

    Returns
    -------
    tables : HashTables
        The buckets of every table, in the compressed layout documented on
        ``HashTables``.
    """
    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    case_indices, n_buckets = _group_cases_by_key(keys)

    table_offsets = np.zeros(n_tables + 1, dtype=np.int64)
    np.cumsum(n_buckets, out=table_offsets[1:])
    total = int(table_offsets[-1])
    bucket_keys = np.empty(total, dtype=np.uint64)
    bucket_starts = np.empty(total, dtype=np.int64)
    _fill_buckets(keys, case_indices, table_offsets, bucket_keys, bucket_starts)
    return HashTables(table_offsets, bucket_keys, bucket_starts, case_indices)


@njit(cache=True)
def _gather_bucket_hits(
    table_offsets, bucket_keys, bucket_starts, case_indices, query_keys
):
    """
    Concatenate the case indices of the bucket the query falls in, per table.

    Parameters
    ----------
    table_offsets : np.ndarray of shape (n_tables + 1,), dtype int64
        Bucket range of each table.
    bucket_keys : np.ndarray of shape (n_buckets,), dtype uint64
        Bucket keys, ascending within each table.
    bucket_starts : np.ndarray of shape (n_buckets,), dtype int64
        Offset of each bucket into its table's row of ``case_indices``.
    case_indices : np.ndarray of shape (n_tables, n_cases), dtype intp
        Case indices grouped by bucket.
    query_keys : np.ndarray of shape (n_tables,), dtype uint64
        The query's bucket key in every table.

    Returns
    -------
    hits : np.ndarray of shape (n_hits,), dtype intp
        The case indices of every probed bucket, table after table. A table
        whose key matches no bucket contributes nothing.
    """
    n_tables, n_cases = case_indices.shape
    starts = np.zeros(n_tables, dtype=np.int64)
    stops = np.zeros(n_tables, dtype=np.int64)
    n_hits = 0
    for t in range(n_tables):
        low = table_offsets[t]
        high = table_offsets[t + 1]
        key = query_keys[t]
        position = low + np.searchsorted(bucket_keys[low:high], key)
        if position < high and bucket_keys[position] == key:
            starts[t] = bucket_starts[position]
            # The last bucket of a table runs to the end of its row.
            if position + 1 < high:
                stops[t] = bucket_starts[position + 1]
            else:
                stops[t] = n_cases
            n_hits += stops[t] - starts[t]

    hits = np.empty(n_hits, dtype=np.intp)
    at = 0
    for t in range(n_tables):
        for r in range(starts[t], stops[t]):
            hits[at] = case_indices[t, r]
            at += 1
    return hits


def _tally_bucket_collisions(tables, keys, n_cases):
    """
    Tally, per colliding case, in how many tables it shares the query's bucket.

    Parameters
    ----------
    tables : HashTables
        The hash tables, as returned by ``_build_hash_tables``.
    keys : np.ndarray of shape (n_tables,)
        The query's integer bucket key in every table.
    n_cases : int
        Number of indexed cases, used to size the dense tally.

    Returns
    -------
    candidates : np.ndarray of shape (n_candidates,), dtype intp
        The distinct case indices that collide with the query in at least one
        table, sorted ascending. Cases that never collide are not listed.
    collisions : np.ndarray of shape (n_candidates,), dtype int
        Collision count of each candidate, between 1 and the number of tables,
        aligned with ``candidates``.
    """
    hits = _gather_bucket_hits(*tables, np.ascontiguousarray(keys, dtype=np.uint64))
    if hits.size == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    # Tally collisions over the concatenated bucket hits with the cheaper of two
    # C-level passes, picked from the number of hits ``h``. Both produce the same
    # (ascending candidates, aligned counts) pair:
    # - dense ``np.bincount``: O(h + n_cases) direct-indexed adds. Wins when the
    #   probed buckets cover a sizeable share of the collection, where sorting
    #   the hits costs up to ~2x the whole query.
    # - sparse ``np.unique``: O(h log h) sort of the hits, independent of
    #   ``n_cases``. Wins when ``h << n_cases``, where the dense pass costs up to
    #   ~13x the tally.
    if hits.size >= n_cases // 8:
        counts = np.bincount(hits, minlength=n_cases)
        candidates = np.flatnonzero(counts)
        collisions = counts[candidates]
    else:
        candidates, collisions = np.unique(hits, return_counts=True)
    return candidates, collisions


def _bucket_dicts(tables):
    """
    Return the hash tables as one ``{bucket key: case indices}`` dict per table.

    The compressed layout is what the index is built and queried through; this
    materializes the same buckets in the shape they are naturally described in,
    for tests and for inspecting a fitted index.

    Parameters
    ----------
    tables : HashTables
        The hash tables, as returned by ``_build_hash_tables``.

    Returns
    -------
    buckets : list of dict
        One dict per table, mapping a bucket key to the ``intp`` array of the
        case indices in that bucket.
    """
    n_cases = tables.case_indices.shape[1]
    out = []
    for t in range(tables.table_offsets.shape[0] - 1):
        low = int(tables.table_offsets[t])
        high = int(tables.table_offsets[t + 1])
        table = {}
        for position in range(low, high):
            start = int(tables.bucket_starts[position])
            stop = int(
                tables.bucket_starts[position + 1] if position + 1 < high else n_cases
            )
            table[int(tables.bucket_keys[position])] = tables.case_indices[
                t, start:stop
            ]
        out.append(table)
    return out
