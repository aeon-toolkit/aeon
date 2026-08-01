"""SimHash (multi-table LSH) index."""

__maintainer__ = ["baraline"]
__all__ = ["SimHashIndexANN"]

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from aeon.distances import get_distance_function, pairwise_distance
from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
from aeon.similarity_search.whole_series._commons import (
    _build_hash_tables,
    _tally_bucket_collisions,
)
from aeon.utils.numba.general import (
    z_normalise_series_2d,
    z_normalise_series_3d,
)
from aeon.utils.validation import check_n_jobs


def _collection_to_signature(X, hash_funcs_flat):
    """
    Compute boolean LSH signatures for a collection of time series.

    The signature of every series is the sign of its dot product with each random
    projection. Flattening series and projections to vectors of length
    ``n_channels * n_timepoints`` turns the whole operation into a single matrix
    product handed to BLAS, which is far faster than a hand-written loop.

    The product runs at ``hash_funcs_flat``'s precision (set from the fitted data's
    dtype): float64 by default, or float32 when the caller passes float32 input,
    which is roughly 2-3x faster and -- since only the sign is kept -- leaves the
    signatures unchanged. ``X`` is cast to that dtype so the matrix product is not
    silently up-cast.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Time series collection to hash.
    hash_funcs_flat : np.ndarray of shape (n_projections, n_channels * n_timepoints)
        Random projection vectors, flattened. Its dtype sets the precision of the
        matrix product.

    Returns
    -------
    res : np.ndarray of shape (n_cases, n_projections)
        Boolean signatures for all time series.
    """
    X_flat = X.reshape(X.shape[0], -1).astype(hash_funcs_flat.dtype, copy=False)
    return (X_flat @ hash_funcs_flat.T) >= 0


def _series_to_signature(X, hash_funcs_flat):
    """
    Compute the boolean LSH signature for a single time series.

    Parameters
    ----------
    X : np.ndarray of shape (n_channels, n_timepoints)
        Time series to hash.
    hash_funcs_flat : np.ndarray of shape (n_projections, n_channels * n_timepoints)
        Random projection vectors, flattened. Its dtype sets the precision of the
        matrix product (see ``_collection_to_signature``).

    Returns
    -------
    res : np.ndarray of shape (n_projections,)
        Boolean signature, one bit per projection.
    """
    x_flat = X.reshape(-1).astype(hash_funcs_flat.dtype, copy=False)
    return (hash_funcs_flat @ x_flat) >= 0


def _signatures_to_keys(signatures, n_tables, n_bits):
    """
    Pack each table's bit-chunk of a signature into a single integer bucket key.

    The ``n_bits`` boolean bits of table ``t`` are read as the binary digits of an
    integer in ``[0, 2 ** n_bits)``. Doing this with a vectorized dot product over
    powers of two replaces the per-series ``tobytes`` calls of the old build loop,
    which is what dominated index construction.

    Parameters
    ----------
    signatures : np.ndarray of shape (n_cases, n_tables * n_bits)
        Boolean LSH signatures.
    n_tables : int
        Number of tables ``L``.
    n_bits : int
        Number of bits ``k`` per table key, at most 64 (the width of the integer
        key).

    Returns
    -------
    keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
        Integer bucket key of every series in every table.
    """
    n_cases = signatures.shape[0]
    chunks = signatures.reshape(n_cases, n_tables, n_bits)
    # Accumulate the key bit-by-bit over the small ``n_bits`` axis rather than
    # materializing an ``(n_cases, n_tables, n_bits)`` uint64 array (and its product)
    # before reducing: peak temporary memory drops by ~n_bits while each bit-plane
    # OR stays a single C-level operation. bit ``b`` contributes ``2 ** b``.
    keys = np.zeros((n_cases, n_tables), dtype=np.uint64)
    for b in range(n_bits):
        keys |= chunks[:, :, b].astype(np.uint64) << np.uint64(b)
    return keys


class SimHashIndexANN(BaseWholeSeriesSearch):
    """
    Approximate nearest neighbor search using multi-table SimHash LSH.

    This is a canonical Locality-Sensitive Hashing (LSH) index for cosine/angular
    similarity. Each series is hashed with **SimHash** (the sign of Gaussian random
    projections over the full series), whose bit-collision probability is exactly
    ``1 - theta / pi`` for the angle ``theta`` between two series.

    The index uses the classic ``(k, L)`` amplification of Indyk-Motwani / Charikar:

    - ``n_bits_per_table`` (``k``) bits are concatenated into a key per table
      (AND-amplification: two series share a bucket only if all ``k`` bits agree),
    - ``n_tables`` (``L``) independent tables are kept (OR-amplification: a series
      is a candidate if it shares the query's bucket in *any* table).

    A query probes its bucket in each of the ``L`` tables and gathers the candidates
    that share at least one of those buckets. ``rerank_distance`` decides how those
    candidates are then ranked:

    - ``None``, the default, ranks them by their **collision count** -- the number of
      tables in which they land in the query's bucket. That count is a cheap proxy for
      angular similarity (a closer series agrees on more bits, so it collides in more
      tables), and the returned distance is its reciprocal ``1 / collision_count``,
      which is monotone in that proxy (smaller means more collisions, i.e. closer).
      Probing a handful of buckets instead of scanning the whole collection, with no
      exact distance computation, is what makes the query sublinear.
    - a distance re-scores the candidates -- and only the candidates -- with it and
      ranks them by that, returning true distances. This is what :class:`SSHIndexANN`
      does unconditionally; it costs one distance computation per candidate and buys
      an exact ordering of a set the collision count can only rank coarsely.

    Note that this method provides **approximate** results: a true neighbor is missed
    only if it never shares a bucket with the query in any table. Ties in the collision
    count are broken arbitrarily (by index) unless ``rerank_distance`` resolves them.
    Larger ``n_tables`` raises recall (and candidate-set size); larger
    ``n_bits_per_table`` makes buckets more selective (smaller candidate sets, faster
    queries, lower recall).

    Parameters
    ----------
    n_tables : int, default=20
        Number of hash tables ``L`` (OR-amplification). More tables increase recall
        and the candidate-set size.
    n_bits_per_table : int, default=8
        Number of bits ``k`` concatenated into each table key (AND-amplification).
        More bits make buckets more selective: smaller candidate sets and faster
        queries, but lower recall.
    rerank_distance : str or callable, default=None
        Distance used to re-rank the candidate set. ``None`` keeps the collision-count
        ranking and its ``1 / collision_count`` proxy distances; set it to score the
        candidates with a real distance instead and return true distances. A list of
        valid strings can be found in the documentation for
        :func:`aeon.distances.get_distance_function`; the name is resolved at fit
        time, so a typo fails before the index is built. There is no
        ``distance_params`` argument: to use non-default parameters, pass a callable
        such as ``partial(dtw_distance, window=0.1)``. The handful of distances whose
        parameters are *required* (``"sax"``, ``"sfa"``, ``"paa_sax"``, ``"dft_sfa"``)
        therefore cannot be given by name, and must be passed configured.
    hash_func_distribution : {"gaussian", "discrete", "uniform"}, default="gaussian"
        Distribution used to draw the random projection vectors. ``"gaussian"`` draws
        from a standard normal, the only choice for which the bit-collision
        probability is exactly ``1 - theta / pi``. ``"discrete"`` draws from
        ``{-1, 1}`` and ``"uniform"`` from ``[-1, 1]``; both approximate the Gaussian
        via the central limit theorem.
    random_state : int, optional
        Random seed for reproducibility of hash function generation.
    normalize : bool, default=True
        Whether to z-normalize series before hashing. Recommended for scale-independent
        matching: the sign random projections then capture angular (cosine) similarity.
        Read at fit time: ``X_`` is stored on the scale it selects, so setting it on a
        fitted estimator has no effect until that estimator is refitted.
    n_jobs : int, default=1
        Number of parallel threads used to hash the collection at fit time and, when
        ``rerank_distance`` is set, for the re-ranking distance computation.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The collection the index searches: the z-normalized collection when
        ``normalize=True``, the raw fitted one otherwise. Storing it on the scale
        queries are compared on is what lets ``rerank_distance`` score candidates by
        indexing straight into it, with no per-query normalization, and what lets it
        be set on an already fitted estimator and still score on the query's scale.
        The default collision-count ranking never reads this attribute at all.
    tables_ : HashTables
        The ``n_tables`` hash tables, holding the case indices of every bucket in
        a flat compressed layout. A bucket key is the ``k`` table bits packed
        into an integer in ``[0, 2 ** k)``. ``_bucket_dicts`` materializes them
        as one ``{bucket key: case indices}`` dict per table.
    hash_funcs_ : np.ndarray of shape (n_tables * n_bits_per_table, n_channels, \
n_timepoints)
        The Gaussian (or discrete/uniform) random projection vectors.
    hash_funcs_flat_ : np.ndarray of shape (n_tables * n_bits_per_table, \
n_channels * n_timepoints)
        ``hash_funcs_`` flattened to one vector per projection, so that hashing is a
        single BLAS matrix product. Its dtype follows the fitted data (float64 by
        default), so fitting on float32 input makes hashing ~2-3x faster without
        changing the (sign-only) signatures.
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.

    Notes
    -----
    In addition to ``k`` and ``axis``, ``predict`` accepts the following search
    option as a keyword argument:

    - ``inverse_distance`` : bool, default=False
        Must be left False. This index captures near neighbors, not far ones, so
        passing ``inverse_distance=True`` raises ``NotImplementedError``. Use
        :class:`NaiveSeriesSearch` with ``inverse_distance=True`` for
        farthest-neighbor queries.

    Unlike :class:`NaiveSeriesSearch`, this estimator does **not** accept
    ``dist_threshold`` or ``X_index``; passing them raises a ``TypeError``.

    ``rerank_distance`` trades query time for ranking quality: the default ranking
    reads no series data at all, while re-ranking scores every candidate. Candidate
    sets grow with ``n_tables`` and shrink with ``n_bits_per_table``, so those two
    parameters set the cost of re-ranking just as much as they set recall.

    See Also
    --------
    SSHIndexANN : LSH index whose collisions are designed to correlate with DTW
        similarity, and which is therefore not defeated by a time shift.
    NaiveSeriesSearch : Exact nearest neighbor search (slower but exact).

    References
    ----------
    .. [1] M. S. Charikar. "Similarity estimation techniques from rounding
       algorithms". STOC 2002. Introduces SimHash (sign random projection) as an
       LSH family for cosine similarity.
    .. [2] P. Indyk and R. Motwani. "Approximate nearest neighbors: towards removing
       the curse of dimensionality". STOC 1998. The multi-table ``(k, L)`` LSH scheme.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.whole_series import SimHashIndexANN
    >>> X_fit = np.random.rand(100, 1, 50)
    >>> query = np.random.rand(1, 50)
    >>> index = SimHashIndexANN()
    >>> index.fit(X_fit)
    SimHashIndexANN()
    >>> indexes, distances = index.predict(query, k=5)

    """

    _tags = {
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        n_tables=20,
        n_bits_per_table=8,
        rerank_distance=None,
        hash_func_distribution="gaussian",
        random_state=None,
        normalize=True,
        n_jobs=1,
    ):
        self.n_tables = n_tables
        self.n_bits_per_table = n_bits_per_table
        self.rerank_distance = rerank_distance
        self.hash_func_distribution = hash_func_distribution
        self.random_state = random_state
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__()

    def _validate_fit_params(self):
        """
        Validate the parameters before the collection is stored and hashed.

        Everything checked here is independent of the fitted data, but running it
        from this hook rather than from ``_fit`` means a bad parameter raises
        before ``fit`` stores ``X_``, so a failed fit leaves nothing of the
        collection attached to the estimator.
        """
        for name, value in (
            ("n_tables", self.n_tables),
            ("n_bits_per_table", self.n_bits_per_table),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {value!r}.")
        if self.n_tables < 1:
            raise ValueError(
                f"n_tables must be a positive integer, got {self.n_tables}."
            )
        if not 1 <= self.n_bits_per_table <= 64:
            raise ValueError(
                "n_bits_per_table must be between 1 and 64 (a table key packs its "
                f"k bits into a 64-bit integer), got {self.n_bits_per_table}."
            )
        if self.hash_func_distribution not in ("gaussian", "discrete", "uniform"):
            raise ValueError(
                "hash_func_distribution must be one of "
                "{'gaussian', 'discrete', 'uniform'}, got "
                f"{self.hash_func_distribution!r}."
            )
        # Resolve ``rerank_distance`` here rather than at the first predict: building
        # the index is the expensive half of this estimator, and a typo that only
        # surfaces once it is paid for wastes exactly what the index is meant to
        # amortise. A callable is returned unchanged.
        if self.rerank_distance is not None:
            # TypeError as well as ValueError: an unhashable argument (a list, say)
            # fails the name lookup with a TypeError that never mentions which
            # parameter was at fault.
            try:
                get_distance_function(self.rerank_distance)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid rerank_distance {self.rerank_distance!r}: {error}"
                ) from error

    def _fit(self, X, y=None):
        """
        Build the multi-table LSH index from X.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            Input data to index and search against the query given to predict.
        y : ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._normalize = self.normalize
        # Hash in the caller's floating precision (float64 by default). Converting
        # the input to float32 therefore speeds up hashing at no cost to the
        # sign-only signatures; see the similarity search example notebook.
        self._input_dtype = (
            X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
        )

        if self._normalize:
            self.X_ = z_normalise_series_3d(X)

        self._initialize_hash_functions()
        # Hashing the collection is a single BLAS matrix product; cap its thread
        # pool to honour n_jobs.
        with threadpool_limits(limits=self._n_jobs, user_api="blas"):
            self._build_index(X)
        return self

    def _initialize_hash_functions(self):
        """Draw the random projection vectors spanning the full series."""
        rng = np.random.default_rng(self.random_state)
        n_projections = self.n_tables * self.n_bits_per_table
        shape = (n_projections, self.n_channels_, self.n_timepoints_)

        if self.hash_func_distribution == "gaussian":
            self.hash_funcs_ = rng.standard_normal(size=shape)
        elif self.hash_func_distribution == "discrete":
            self.hash_funcs_ = rng.choice([-1, 1], size=shape)
        else:
            # "uniform", the only value left once ``_validate_fit_params`` has run.
            self.hash_funcs_ = rng.uniform(low=-1, high=1.0, size=shape)

        # Flatten to one vector per projection in the fitted data's precision:
        # hashing is then a single BLAS matrix product (see
        # ``_collection_to_signature``). Keeping the input dtype means float32 input
        # makes hashing ~2-3x faster while leaving the sign-only signatures intact.
        self.hash_funcs_flat_ = self.hash_funcs_.reshape(n_projections, -1).astype(
            self._input_dtype, copy=False
        )

    def _build_index(self, X):
        """
        Hash the collection and populate the ``n_tables`` hash tables.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Time series collection to index.
        """
        signatures = _collection_to_signature(X, self.hash_funcs_flat_)
        keys = _signatures_to_keys(signatures, self.n_tables, self.n_bits_per_table)
        self.tables_ = _build_hash_tables(keys, self.n_tables)

    def _predict(self, X, k=1, inverse_distance=False):
        """
        Find the k approximate nearest neighbors for a query series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series.
        k : int, optional
            Number of neighbors to return. Default is 1.
        inverse_distance : bool, default=False
            Not supported by a near-neighbor bucket index. Must be left False;
            passing True raises ``NotImplementedError``. Use
            :class:`NaiveSeriesSearch` with ``inverse_distance=True`` for
            farthest-neighbor queries.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Indices of the neighbor series in the database, most likely neighbor
            first: by decreasing collision count when ``rerank_distance`` is None, by
            increasing ``rerank_distance`` otherwise. ``n_found`` may be smaller than
            ``k`` if too few candidates collide with the query.
        distances : np.ndarray of shape (n_found,)
            With ``rerank_distance=None``, the proxy distances
            ``1 / collision_count``, smaller meaning the neighbor collided in more
            tables. Otherwise, the true distances of the returned neighbors to the
            query under ``rerank_distance``.
        """
        if inverse_distance:
            raise NotImplementedError(
                "SimHashIndexANN does not support inverse_distance: its "
                "buckets capture near neighbors, not far ones. Use "
                "NaiveSeriesSearch with inverse_distance=True for "
                "farthest-neighbor queries."
            )
        self._check_query_length(X)

        if self._normalize:
            X = z_normalise_series_2d(X)

        if k == np.inf:
            # "Return every match": clamping is the documented meaning here, not
            # a user mistake, so it must not warn.
            k = self.n_cases_
        elif k > self.n_cases_:
            warnings.warn(
                f"k={k} is larger than the number of indexed cases "
                f"({self.n_cases_}). Returning at most {self.n_cases_} neighbors.",
                UserWarning,
                stacklevel=3,
            )
            k = self.n_cases_

        candidates, collisions = self._gather_candidates(X)
        if self.rerank_distance is None:
            return self._rank_candidates(candidates, collisions, k)
        # The collision count only selected the candidates here; the ranking comes
        # from the distance instead.
        return self._rerank_candidates(X, candidates, k)

    def _gather_candidates(self, X):
        """
        Tally, per colliding case, in how many tables it shares the query's bucket.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series (already normalized if ``normalize`` is True).

        Returns
        -------
        candidates : np.ndarray of shape (n_candidates,), dtype intp
            The distinct case indices that collide with the query in at least one
            table, sorted ascending. Cases that never collide are not listed.
        collisions : np.ndarray of shape (n_candidates,), dtype int
            Collision count of each candidate: the number of tables in which it lands
            in the query's bucket, between 1 and ``n_tables``. Aligned with
            ``candidates``.
        """
        signature = _series_to_signature(X, self.hash_funcs_flat_)
        keys = _signatures_to_keys(
            signature[None, :], self.n_tables, self.n_bits_per_table
        )[0]
        return _tally_bucket_collisions(self.tables_, keys, self.n_cases_)

    def _rank_candidates(self, candidates, collisions, k):
        """
        Rank candidates by collision count and keep the top k.

        Parameters
        ----------
        candidates : np.ndarray of shape (n_candidates,)
            Distinct candidate case indices (sorted ascending), as returned by
            ``_gather_candidates``.
        collisions : np.ndarray of shape (n_candidates,)
            Collision count of each candidate, aligned with ``candidates``.
        k : int
            Number of neighbors to return.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Top-k candidate indices ordered by decreasing collision count (ties broken
            by ascending index for determinism).
        distances : np.ndarray of shape (n_found,)
            The proxy distances ``1 / collision_count`` for the returned neighbors.
        """
        if len(candidates) == 0:
            warnings.warn(
                "No candidates collided with the query in any table; returning no "
                "neighbors. Increase n_tables or decrease n_bits_per_table.",
                UserWarning,
                stacklevel=3,
            )
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        # primary key: collision count descending; tie-break: index ascending.
        # ``candidates`` is already ascending, so this matches the previous
        # ``np.lexsort((indexes, -collisions))`` over the dense tally.
        order = np.lexsort((candidates, -collisions))
        n_found = min(k, len(candidates))
        order = order[:n_found]

        if n_found < k:
            warnings.warn(
                f"Only {n_found} candidates collided with the query, fewer than the "
                f"requested k={k}. Increase n_tables or decrease n_bits_per_table.",
                UserWarning,
                stacklevel=3,
            )
        return candidates[order], 1.0 / collisions[order]

    def _rerank_candidates(self, X, candidates, k):
        """
        Score the candidate set with ``rerank_distance`` and keep the top k.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series (already normalized if ``normalize`` is True).
        candidates : np.ndarray of shape (n_candidates,)
            Distinct candidate case indices (sorted ascending), as returned by
            ``_gather_candidates``.
        k : int
            Number of neighbors to return.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Top-k candidate indices ordered by increasing distance (ties broken by
            ascending index for determinism).
        distances : np.ndarray of shape (n_found,)
            The true distances under ``rerank_distance``, aligned with ``indexes``.
        """
        if len(candidates) == 0:
            warnings.warn(
                "No candidates collided with the query in any table; returning no "
                "neighbors. Increase n_tables or decrease n_bits_per_table.",
                UserWarning,
                stacklevel=3,
            )
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        # Score the candidates only: this is what keeps the query sublinear while
        # still returning true distances.
        distances = pairwise_distance(
            self.X_[candidates],
            X[np.newaxis],
            method=self.rerank_distance,
            n_jobs=self._n_jobs,
        ).reshape(-1)
        # primary key: distance ascending; tie-break: index ascending.
        order = np.lexsort((candidates, distances))
        n_found = min(k, len(candidates))
        order = order[:n_found]

        if n_found < k:
            warnings.warn(
                f"Only {n_found} candidates collided with the query, fewer than the "
                f"requested k={k}. Increase n_tables or decrease n_bits_per_table.",
                UserWarning,
                stacklevel=3,
            )
        return candidates[order], distances[order]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """
        Return testing parameter settings for the estimator.

        Two settings are returned so that aeon's check harness covers both
        rankings: the default collision count and a re-ranking distance.
        """
        if parameter_set == "default":
            return [
                {"n_tables": 4, "n_bits_per_table": 4},
                {"n_tables": 4, "n_bits_per_table": 4, "rerank_distance": "dtw"},
            ]
        raise NotImplementedError(
            f"The parameter set {parameter_set} is not yet implemented"
        )
