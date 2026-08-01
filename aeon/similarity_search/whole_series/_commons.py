"""Shared building blocks for whole series hash indexes."""

__maintainer__ = ["baraline"]

import numpy as np


def _build_hash_tables(keys, n_tables):
    """
    Group case indices into per-table buckets from their integer bucket keys.

    Buckets are built with a stable sort instead of a per-case Python dict loop
    (which was ``n_cases * n_tables`` interpreter-level ops). A stable argsort
    groups equal keys while preserving ascending case order within each group,
    so ``np.split`` at the unique-key boundaries yields buckets whose index
    arrays are identical to the dict-insert order. Each bucket is an int array
    so a query can tally collisions with a single C-level pass over the
    concatenated buckets (see ``_tally_bucket_collisions``).

    Parameters
    ----------
    keys : np.ndarray of shape (n_cases, n_tables)
        Integer bucket key of every case in every table.
    n_tables : int
        Number of hash tables.

    Returns
    -------
    tables : list of dict
        One dict per table, mapping a bucket key to the ``intp`` array of the
        case indices that fall in that bucket.
    """
    tables = []
    for t in range(n_tables):
        col = keys[:, t]
        order = np.argsort(col, kind="stable")
        unique_keys, first_index = np.unique(col[order], return_index=True)
        buckets = np.split(order.astype(np.intp), first_index[1:])
        tables.append(dict(zip(unique_keys.tolist(), buckets)))
    return tables


def _tally_bucket_collisions(tables, keys, n_cases):
    """
    Tally, per colliding case, in how many tables it shares the query's bucket.

    Parameters
    ----------
    tables : list of dict
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
        Collision count of each candidate, between 1 and ``len(tables)``,
        aligned with ``candidates``.
    """
    hit_arrays = []
    for table, key in zip(tables, keys, strict=True):
        bucket = table.get(int(key))
        if bucket is not None:
            hit_arrays.append(bucket)
    if len(hit_arrays) == 0:
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
    hits = np.concatenate(hit_arrays)
    if hits.size >= n_cases // 8:
        counts = np.bincount(hits, minlength=n_cases)
        candidates = np.flatnonzero(counts)
        collisions = counts[candidates]
    else:
        candidates, collisions = np.unique(hits, return_counts=True)
    return candidates, collisions
