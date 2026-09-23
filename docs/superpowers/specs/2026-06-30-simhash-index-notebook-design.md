# Design: SimHash deep-dive notebook

## Goal

Create a new example notebook that explains the **mechanics of SimHash and its use
as an approximate nearest-neighbour index** in aeon's `similarity_search` module.
It is the SimHash counterpart to
[`distance_profiles.ipynb`](../../../examples/similarity_search/distance_profiles.ipynb),
which explains the theory and math behind distance profiles. Where that notebook
is the "deep dive" for the *subsequence/exact* side, this one is the deep dive for
the *whole-series/approximate* (`SimHashIndexANN`) side.

## Context

The `similarity_search` example folder currently has three notebooks:

- `similarity_search.ipynb` — module overview; already contains a short prose
  introduction to `SimHashIndexANN`.
- `distance_profiles.ipynb` — pure-markdown math deep dive (the style/voice
  reference for this work).
- `code_speed.ipynb` — performance and speed-ups; already covers the **empirical
  tuning** of SimHash (recall/purity/speed vs `n_tables` and `n_bits_per_table`,
  and the float32 vs float64 trade-off).

The implementation lives in
[`aeon/similarity_search/whole_series/_simhash_index_ann.py`](../../../aeon/similarity_search/whole_series/_simhash_index_ann.py).
Key facts the notebook draws on:

- A bit is `1[⟨w, x⟩ ≥ 0]` for a random projection vector `w`; series are flattened
  to `n_channels * n_timepoints` so hashing the whole collection is a single BLAS
  matmul `X_flat @ hash_funcs_flat_.T >= 0` (`_collection_to_signature`).
- Bit-collision probability between two series at angle θ is exactly `1 - θ/π`
  (gaussian projections; Charikar 2002).
- `(k, L)` amplification: `n_bits_per_table` (`k`) bits AND-ed into one table key,
  `n_tables` (`L`) tables OR-ed together (Indyk–Motwani 1998).
- `_signatures_to_keys` packs each table's `k` bits into an integer bucket key via
  powers of two; `tables_` is a list of `dict` mapping key → list of case indices.
- A query is ranked by **collision count** (number of tables whose bucket it
  shares with the query); reported proxy distance is `1 / collision_count`.
- `normalize=True` z-normalizes series so the sign projections capture angular
  (cosine) similarity.
- `hash_func_distribution` ∈ {gaussian, discrete, uniform}; gaussian is the only
  one for which `1 - θ/π` is exact, the other two approximate it via the CLT.

## Decisions (resolved during brainstorming)

1. **Format:** markdown math derivations interleaved with runnable code and
   matplotlib plots (not pure-markdown like `distance_profiles.ipynb`). SimHash is
   geometric, so visualizing it is worth the extra cells.
2. **Scope:** theory **and** engineering. Explain the LSH math *and* how aeon
   implements it as an index (BLAS matmul, bit-packing into integer keys, hash
   table build/probe). **Do not** reproduce the empirical tuning curves — those
   stay in `code_speed.ipynb`; link to it instead.
3. **Demo data:** simple 2D synthetic points/vectors for the geometry (hyperplanes,
   sign regions, angle→collision), then a real aeon dataset (`load_arrow_head`, as
   the other notebooks use) for the actual index build/probe demo.
4. **Tone:** match `distance_profiles.ipynb` — clear, explanatory "we" prose,
   intuition stated plainly, math shown step by step. Accessible but technical; not
   jokey or overly casual.

## Notebook structure

File: `examples/similarity_search/simhash_index.ipynb`.

Each section is a markdown explanation; sections 2, 4, 5, 6, 7 also have code +
plots as noted.

1. **What is SimHash, and why use it for search?** *(markdown)*
   Open like `distance_profiles.ipynb` ("In this notebook, we will..."). Recap that
   exact whole-series search compares the query against every series — O(n_cases)
   distance computations per query — which gets slow on large collections. Introduce
   the LSH idea: give each series a short fingerprint such that similar series tend
   to share it, then at query time only compare against series sharing the query's
   fingerprint. Forward-link to the overview and code_speed notebooks.

2. **Hashing a series with a random projection** *(markdown + plot)*
   A single random vector `w` defines a hyperplane through the origin; the hash bit
   is which side a series falls on, `b = 1[⟨w, x⟩ ≥ 0]`. Explain flattening a
   `(n_channels, n_timepoints)` series into one vector so the projection is a dot
   product. **Plot:** 2D synthetic points, draw the hyperplane (normal `w`), color
   points by their bit; show that the bit is just the side of the line.

3. **Why we normalize before hashing** *(markdown, optional tiny plot)*
   `sign(⟨w, x⟩)` depends only on the *direction* of `x`, not its magnitude.
   z-normalizing (zero mean, unit variance) removes offset and scale so the bit
   captures shape/angle — i.e. cosine/angular similarity. Tie to `normalize=True`.

4. **The collision probability of two series** *(markdown + plot)*
   Derive, conversationally, why two series at angle θ collide on one bit with
   probability `1 - θ/π`: a random hyperplane's normal is uniformly oriented; it
   separates the two vectors iff it falls in the "wedge" of angle θ between them,
   which happens with probability `θ/π`. **Plot:** for several fixed angles, draw
   many random projections, measure the empirical collision rate, and overlay the
   `1 - θ/π` line to confirm.

5. **Amplification with bits and tables (k and L)** *(markdown + plot)*
   One bit is a weak hash (far series still collide ~half the time). Two knobs fix
   this:
   - **AND within a table:** concatenate `k` bits; two series share a table key only
     if all `k` bits agree, so collision probability becomes `(1 - θ/π)^k` — buckets
     get selective.
   - **OR across tables:** keep `L` independent tables; a series is a candidate if it
     collides in *any* table, giving `1 - (1 - (1 - θ/π)^k)^L`.
   **Plot:** the classic LSH **S-curve** — probability a series is retrieved as a
   candidate vs θ (or cosine similarity) for a few `(k, L)` settings — showing how
   `k` sharpens the threshold and `L` raises the curve. (This figure is also reused
   as the docs cover image; see Integration.)

6. **Building the index: from signatures to buckets** *(markdown + code)*
   The engineering. On a real `load_arrow_head` collection:
   - Hashing the whole collection is one BLAS matmul `X_flat @ W.T >= 0`
     (`_collection_to_signature`); show the boolean signature matrix shape.
   - Packing a table's `k` bits into an integer key via powers of two
     (`_signatures_to_keys`); show a few real keys.
   - Building `tables_`: one `dict` per table, key → list of case indices; inspect a
     bucket. Reference the real fitted attributes (`hash_funcs_flat_`, `tables_`).

7. **Answering a query** *(markdown + code/plot)*
   The query path: hash the query, probe its bucket in each of `L` tables, gather the
   union of candidates, rank by **collision count**, return top-k with proxy distance
   `1 / collision_count`. Run the real `SimHashIndexANN` on `arrow_head`. **Plot/show:**
   candidate-set size vs full collection size (the sublinear win), and compare the
   approximate neighbours to exact `BruteForce` neighbours. State the approximate
   nature plainly: a true neighbour is missed if it never shares a bucket; ties in
   collision count are broken by index.

8. **The choice of random projection distribution** *(short markdown)*
   `gaussian` is rotationally symmetric, which is what makes `1 - θ/π` exact;
   `discrete` `{-1, 1}` and `uniform` `[-1, 1]` approximate the gaussian via the
   central limit theorem (each projection sums many terms). Kept brief.

9. **Conclusion and further reading** *(markdown)*
   Recap the mechanics in a sentence or two; link to `code_speed.ipynb` for tuning
   `n_tables`/`n_bits_per_table` and the float32/float64 trade-off, and to the
   overview notebook. Cite Charikar 2002 and Indyk–Motwani 1998 (already in the
   estimator docstring).

## Integration

- **docs/examples.md:** add a 4th grid-item-card under the "Similarity Search"
  section (after the code_speed card), titled "Deep dive into SimHash and the LSH
  index", linking `/examples/similarity_search/simhash_index.ipynb`.
- **Cover image:** generate `examples/similarity_search/img/simhash_index.png` from
  the notebook's own S-curve plot (section 5), to match the existing three cards
  which all use an `img/*.png`.
- **Cross-links:** update the "Where to next?" / "Other similarity search notebooks"
  lists in `similarity_search.ipynb` and `distance_profiles.ipynb` to include the new
  notebook.

## Verification

- Execute the notebook end-to-end with the repo's local `.venv` python (per project
  convention — not conda) so all outputs are populated and it runs cleanly with no
  errors.
- Use `random_state` everywhere randomness appears so the notebook is reproducible.
- Confirm the new docs card renders the same shape as the existing three (image +
  title + link), and that the cover PNG exists at the referenced path.
- Lint any code cells in line with repo style (ruff/isort/black), consistent with
  the existing notebooks.

## Out of scope

- Empirical tuning benchmarks (recall/purity/speed sweeps, float32 vs float64
  timings) — these remain in `code_speed.ipynb`.
- Any change to the `SimHashIndexANN` implementation itself; this is documentation
  only.
- A `rerank` feature (noted as future work in `code_speed.ipynb`); only mention it
  if it naturally comes up, without implying it exists.
