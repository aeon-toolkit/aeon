# Design: `predict_batch` for whole-series similarity search

Date: 2026-07-01
Component: `aeon.similarity_search.whole_series`

## Goal

Add a `predict_batch` method to `BaseWholeSeriesSearch` that predicts nearest
neighbors for a **collection** of query series. By default the method reuses the
child class's single-query `_predict`, but a child can override a private hook to
provide a batch-specific optimization.

## Motivation

Callers frequently have many queries to run against the same fitted index (e.g.
k-NN classification of a whole test collection). Running them one at a time
through `predict` repeats per-query Python overhead and, for index/hash based
estimators, misses the chance to vectorize the expensive step. For
`SimHashIndexANN` in particular, hashing is a single BLAS matrix product that is
far cheaper amortized over the whole batch than per query.

## Design decisions (settled during brainstorming)

1. **Override mechanism**: public `predict_batch` (validation/preprocessing) →
   private `_predict_batch` hook, matching the existing `fit`/`_fit`,
   `predict`/`_predict` split. The base `_predict_batch` is *concrete* (a default
   loop), not abstract, so every child works out of the box.
2. **Return format**: two Python lists, `(indexes, distances)`, each of length
   `n_queries`; element `i` is that query's 1D array (ragged, possibly fewer than
   `k`), exactly what `predict` returns for query `i`.
3. **Input handling**: 3D numpy array `(n_queries, n_channels, n_timepoints)` with
   an `axis` parameter mirroring `predict`. Preprocessed once via
   `_preprocess_collection(store_metadata=False)` so fit metadata is preserved.
4. **Scope**: deliver the base method + hook, plus one real optimized override for
   `SimHashIndexANN`. No batched `BruteForce` override in this task.

## Component 1 — Public API: `BaseWholeSeriesSearch.predict_batch`

```python
def predict_batch(self, X, k=1, axis=1, **kwargs):
    # -> (indexes, distances), each a list of length n_queries
```

Behavior:

- `X`: a 3D numpy array `(n_queries, n_channels, n_timepoints)`.
- `self._check_is_fitted()`.
- Validate `X` is an `np.ndarray` with `ndim == 3`; otherwise raise `ValueError`
  with a clear message.
- If `axis == 0`, interpret `X` as `(n_queries, n_timepoints, n_channels)` and
  transpose the last two axes to the standard form (mirrors `predict`'s axis
  semantics at the collection level).
- Preprocess **once**: `X = self._preprocess_collection(X, store_metadata=False)`
  (converts to the numpy3D inner type, checks capabilities; does not overwrite
  fit metadata).
- Single shape check on the rectangular collection: `X.shape[1] == self.n_channels_`
  and `X.shape[2] == self.n_timepoints_`, else `ValueError`. Doing both once here
  keeps the override path safe (the `SimHash` override skips `_predict`, hence
  skips `_check_query_length`); the default loop still re-checks length per query
  inside each child's `_predict`, which is harmless.
- `indexes, distances = self._predict_batch(X, k, **kwargs)`.
- Return `(indexes, distances)`.

Return contract: `indexes` and `distances` are each a list of length
`n_queries`; `indexes[i]`, `distances[i]` are the 1D arrays that
`predict(X[i], k, ...)` would have returned.

## Component 2 — Default hook: `BaseWholeSeriesSearch._predict_batch`

```python
def _predict_batch(self, X, k, **kwargs):
    indexes, distances = [], []
    for i in range(X.shape[0]):
        idx, dist = self._predict(X[i], k, **kwargs)
        indexes.append(idx)
        distances.append(dist)
    return indexes, distances
```

- Concrete method (default), overridable by children.
- Loops the child's own `_predict`, so results are identical to calling
  `predict` per query (minus the repeated collection preprocessing, which the
  public method already did once).
- `**kwargs` are passed uniformly to every query.

## Component 3 — Optimized override: `SimHashIndexANN._predict_batch`

Hashing the queries is the expensive step and is a single BLAS matmul across the
whole batch. Steps:

1. If `normalize`, `X = z_normalise_series_3d(X)`.
2. Cap `k` to `self.n_cases_` **once** (a single warning for the whole batch,
   instead of one per query).
3. `signatures = _collection_to_signature(X, self.hash_funcs_flat_)`
   → shape `(n_queries, n_projections)` — one matmul for all queries.
4. `keys = _signatures_to_keys(signatures, self.n_tables, self.n_bits_per_table)`
   → shape `(n_queries, n_tables)`.
5. For each query `i`: gather bucket hits from `keys[i]` and tally with
   `np.bincount` → `counts`, then rank top-k.
6. Return `(indexes_list, distances_list)`.

### Supporting refactor (no behavior change to single-query path)

- Extract the "gather collision counts from precomputed table keys" portion of
  `_gather_candidates` into a helper `_counts_from_keys(keys_row)` that takes a
  1D array of `n_tables` integer keys and returns the `counts` array.
  `_gather_candidates(X)` keeps computing the signature/keys and then calls
  `_counts_from_keys`; its behavior is unchanged.
- Add a `warn=True` parameter to `_rank_candidates(counts, k, warn=True)`. The
  batch path passes `warn=False` to **suppress the per-query "no candidates" /
  "fewer than k candidates" warnings**, which would otherwise spam over a large
  batch. This is consistent with the "validate/warn once" philosophy. Single
  `_predict` keeps `warn=True` (unchanged behavior).

Results (indexes/distances) from the override are identical to looping `predict`;
only warning verbosity differs.

## Error handling

- Not fitted → the standard not-fitted error via `_check_is_fitted()`.
- `X` not an `np.ndarray` or not 3D → `ValueError` naming the expected shape.
- Wrong channel count or wrong timepoint length → `ValueError` from the single
  shape check in `predict_batch`, naming expected vs got. (The default loop also
  re-checks length inside child `_predict` via `_check_query_length`, unchanged.)

## Testing (TDD)

- **Base default equivalence**: for `BruteForce`, `predict_batch` on a stacked
  set of queries equals calling `predict` on each query individually (compare
  both `indexes` and `distances`, allowing for the ragged list structure).
- **SimHash override equivalence**: `SimHashIndexANN.predict_batch` results equal
  looped `predict` (compare arrays; ignore warnings). Confirms the override is a
  pure optimization, not a behavior change.
- **Return contract**: result is two lists, each of length `n_queries`; ragged /
  fewer-than-k per-query results are preserved.
- **Axis path**: `predict_batch(X, axis=0)` matches `predict_batch(X_t, axis=1)`
  where `X_t` is `X` with the last two axes transposed.
- **Errors**: non-3D input and wrong channel count raise `ValueError` with clear
  messages.
- **Warning suppression**: a batch that triggers "fewer than k candidates" for
  some queries does not emit one warning per query in the SimHash override.

## Out of scope / limitations

- Per-query self-exclusion (`X_index` per query) is not supported by the batch
  path; `**kwargs` are applied uniformly to all queries. Noted as a limitation;
  callers needing per-query exclusion should use `predict` in a loop.
- No batched `BruteForce` override in this task (its per-query top-k already
  parallelizes over cases via numba).
