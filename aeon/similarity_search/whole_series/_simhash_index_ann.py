"""SimHash (multi-table LSH) index."""

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
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
    powers = np.uint64(1) << np.arange(n_bits, dtype=np.uint64)
    return (chunks.astype(np.uint64) * powers).sum(axis=2)


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
    that share at least one of those buckets. Candidates are ranked by their
    **collision count** -- the number of tables in which they land in the query's
    bucket -- and the top ``k`` are returned. The collision count is a cheap proxy for
    angular similarity (a closer series agrees on more bits, so it collides in more
    tables); the returned distance is its reciprocal ``1 / collision_count``, which is
    monotone in that proxy (smaller means more collisions, i.e. closer). Probing a
    handful of buckets instead of scanning the whole collection, with no exact distance
    computation, is what makes the query sublinear.

    Note that this method provides **approximate** results: a true neighbor is missed
    only if it never shares a bucket with the query in any table, and ties in the
    collision count are broken arbitrarily (by index). Larger ``n_tables`` raises recall
    (and candidate-set size); larger ``n_bits_per_table`` makes buckets more selective
    (smaller candidate sets, faster queries, lower recall).

    Parameters
    ----------
    n_tables : int, default=20
        Number of hash tables ``L`` (OR-amplification). More tables increase recall
        and the candidate-set size.
    n_bits_per_table : int, default=8
        Number of bits ``k`` concatenated into each table key (AND-amplification).
        More bits make buckets more selective: smaller candidate sets and faster
        queries, but lower recall.
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
    n_jobs : int, default=1
        Number of parallel threads used to hash the collection at fit time.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection of time series (as stored by the base class).
    tables_ : list of dict
        The ``n_tables`` hash tables, each mapping a ``k``-bit bucket key (the
        ``k`` table bits packed into an integer in ``[0, 2 ** k)``) to an int array
        of the case indices that fall in that bucket.
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

    See Also
    --------
    BruteForce : Exact nearest neighbor search (slower but exact).

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
        hash_func_distribution="gaussian",
        random_state=None,
        normalize=True,
        n_jobs=1,
    ):
        self.n_tables = n_tables
        self.n_bits_per_table = n_bits_per_table
        self.hash_func_distribution = hash_func_distribution
        self.random_state = random_state
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__()

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
        if not 1 <= self.n_bits_per_table <= 64:
            raise ValueError(
                "n_bits_per_table must be between 1 and 64 (a table key packs its "
                f"k bits into a 64-bit integer), got {self.n_bits_per_table}."
            )
        # Hash in the caller's floating precision (float64 by default). Converting
        # the input to float32 therefore speeds up hashing at no cost to the
        # sign-only signatures; see the similarity search example notebook.
        self._input_dtype = (
            X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
        )

        if self.normalize:
            X = z_normalise_series_3d(X)

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
        elif self.hash_func_distribution == "uniform":
            self.hash_funcs_ = rng.uniform(low=-1, high=1.0, size=shape)
        else:
            raise ValueError(
                "hash_func_distribution must be one of "
                "{'gaussian', 'discrete', 'uniform'}, got "
                f"{self.hash_func_distribution!r}."
            )

        # Flatten to one vector per projection in the fitted data's precision:
        # hashing is then a single BLAS matrix product (see
        # ``_collection_to_signature``). Keeping the input dtype means float32 input
        # makes hashing ~2-3x faster while leaving the sign-only signatures intact.
        self.hash_funcs_flat_ = self.hash_funcs_.reshape(n_projections, -1).astype(
            self._input_dtype
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

        # One pass per table over precomputed integer keys: a plain dict insert per
        # series, with no per-series ``tobytes`` or slicing (which dominated the old
        # build loop). ``.tolist()`` hands the loop native Python ints.
        self.tables_ = []
        for t in range(self.n_tables):
            table = {}
            for i, key in enumerate(keys[:, t].tolist()):
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [i]
                else:
                    bucket.append(i)
            # Freeze each bucket's index list into an int array so a query can tally
            # collisions with a single ``np.bincount`` over the concatenated buckets
            # instead of a per-candidate Python loop (see ``_gather_candidates``).
            self.tables_.append(
                {key: np.asarray(idxs, dtype=np.intp) for key, idxs in table.items()}
            )

    def _predict(self, X, k=1, inverse_distance=False):
        """
        Find the k approximate nearest neighbors for a query series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series.
        k : int, optional
            Number of neighbors to return. Default is 1.
        inverse_distance : bool, optional
            Not supported by a near-neighbor bucket index. Default is False.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Indices of the neighbor series in the database, ordered by decreasing
            collision count (most likely neighbor first). ``n_found`` may be smaller
            than ``k`` if too few candidates collide with the query.
        distances : np.ndarray of shape (n_found,)
            Proxy distances ``1 / collision_count`` for the returned neighbors;
            smaller means the neighbor collided in more tables.
        """
        if inverse_distance:
            raise NotImplementedError(
                "SimHashIndexANN does not support inverse_distance: its "
                "buckets capture near neighbors, not far ones. Use BruteForce with "
                "inverse_distance=True for farthest-neighbor queries."
            )
        self._check_query_length(X)

        if self.normalize:
            X = z_normalise_series_2d(X)

        if k > self.n_cases_:
            warnings.warn(
                f"k={k} is larger than the number of indexed cases "
                f"({self.n_cases_}). Returning at most {self.n_cases_} neighbors.",
                UserWarning,
                stacklevel=3,
            )
            k = self.n_cases_

        counts = self._gather_candidates(X)
        return self._rank_candidates(counts, k)

    def _gather_candidates(self, X):
        """
        Count, for each case, in how many tables it shares the query's bucket.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series (already normalized if ``normalize`` is True).

        Returns
        -------
        counts : np.ndarray of shape (n_cases_,), dtype int
            Collision count of every case: the number of tables in which it lands in
            the query's bucket, between 0 and ``n_tables``. Cases that never collide
            with the query have a count of 0.
        """
        signature = _series_to_signature(X, self.hash_funcs_flat_)
        keys = _signatures_to_keys(
            signature[None, :], self.n_tables, self.n_bits_per_table
        )[0]
        # Concatenate the case indices of every probed bucket and tally collisions
        # with a single C-level ``np.bincount``. This replaces the per-candidate
        # Python dict-increment loop that dominated query time whenever the candidate
        # set was large (few bits / many tables), where it was slower than brute force.
        hit_arrays = []
        for t in range(self.n_tables):
            bucket = self.tables_[t].get(int(keys[t]))
            if bucket is not None:
                hit_arrays.append(bucket)
        if len(hit_arrays) == 0:
            return np.zeros(self.n_cases_, dtype=np.intp)
        return np.bincount(np.concatenate(hit_arrays), minlength=self.n_cases_)

    def _rank_candidates(self, counts, k):
        """
        Rank candidates by collision count and keep the top k.

        Parameters
        ----------
        counts : np.ndarray of shape (n_cases_,)
            Collision count of every case (0 for cases that never collided with the
            query), as returned by ``_gather_candidates``.
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
        indexes = np.nonzero(counts)[0]
        if len(indexes) == 0:
            warnings.warn(
                "No candidates collided with the query in any table; returning no "
                "neighbors. Increase n_tables or decrease n_bits_per_table.",
                UserWarning,
                stacklevel=3,
            )
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        collisions = counts[indexes]
        # primary key: collision count descending; tie-break: index ascending
        order = np.lexsort((indexes, -collisions))
        n_found = min(k, len(indexes))
        order = order[:n_found]

        if n_found < k:
            warnings.warn(
                f"Only {n_found} candidates collided with the query, fewer than the "
                f"requested k={k}. Increase n_tables or decrease n_bits_per_table.",
                UserWarning,
                stacklevel=3,
            )
        return indexes[order], 1.0 / collisions[order]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator."""
        if parameter_set == "default":
            return {"n_tables": 4, "n_bits_per_table": 4}
        raise NotImplementedError(
            f"The parameter set {parameter_set} is not yet implemented"
        )
