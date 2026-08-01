"""SSH (Sketch, Shingle & Hash) index."""

__maintainer__ = ["baraline"]
__all__ = ["SSHIndexANN"]

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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

# Target size, in bytes, of the transient arrays the whole SSH pipeline holds at
# once. ``_hash_chunk_size`` turns it into a number of cases and
# ``SSHIndexANN._hash_collection`` hashes the collection in chunks of that size,
# which is what keeps peak memory independent of ``n_cases``.
#
# This is a working-set target, not just a ceiling: it is set well below what a
# modern machine could hold because the pipeline is memory-bound, and a chunk whose
# arrays stay in cache is markedly faster. Fitting 5000 ECG5000 series
# (n_tables=40, k=8) takes 434 ms at 8 MiB against 1534 ms at 64 MiB, and 1005 ms
# against 3394 ms for the longer sketch of ``shift=1`` -- 1.6x to 3.8x, with peak
# memory 20 MB against 66 MB. Raising it costs time as well as memory.
_HASH_CHUNK_BYTES = 8 * 1024 * 1024

# Number of ``(n_chunk, n_shingles)`` 8-byte arrays that ``_hash_chunk_size``
# budgets for; see its docstring for how the count is derived from the code.
_LIVE_SHINGLE_ARRAYS = 7


def _n_sketch_bits(n_timepoints, window_length, shift):
    """
    Return the number of sketch bits produced for a series.

    Parameters
    ----------
    n_timepoints : int
        Length of the series.
    window_length : int
        Length ``W`` of the random filter.
    shift : int
        Step size ``delta`` between two consecutive filter positions.

    Returns
    -------
    n_bits : int
        ``floor((n_timepoints - window_length) / shift) + 1``, the number of
        filter positions that fit in the series.
    """
    return (n_timepoints - window_length) // shift + 1


def _hash_chunk_size(n_bits, n_shingles, n_channels, window_length, itemsize):
    """
    Return how many cases the SSH pipeline may hash in a single pass.

    Every stage of the pipeline is row-independent, so a collection can be hashed
    in chunks of rows, and peak memory -- and the working set the pipeline sweeps
    per stage -- then depends on the chunk size instead of on ``n_cases``. Two
    families of allocation are budgeted for, per case:

    - the strided window block that ``_collection_to_sketch`` copies before its
      matrix product, ``n_bits * n_channels * window_length * itemsize`` bytes;
    - the ``(n_chunk, n_shingles)`` 8-byte arrays of the stages after it. Six of
      them are live at the same time at the worst point, which is inside
      ``_occurrence_ranks``: ``ids`` (held by the caller), ``order``,
      ``sorted_ids``, ``group_start``, ``ranks_sorted`` and ``ranks``.
      ``_LIVE_SHINGLE_ARRAYS`` is one above that count, as headroom for the
      boolean ``new_group`` (an eighth of the size) and for the indexing
      temporaries of ``np.put_along_axis``. Measured with ``tracemalloc``, the
      pipeline peaks at 6.3 times ``n_shingles * 8`` bytes per case.

    The two families are never live simultaneously -- the window block is freed
    when ``_collection_to_sketch`` returns, before shingling starts -- so adding
    them errs on the side of a smaller chunk than the budget would allow.

    Parameters
    ----------
    n_bits : int
        Length of the sketch bit string, ``_n_sketch_bits(...)``.
    n_shingles : int
        Number of shingle occurrences per case, ``n_bits - shingle_size + 1``.
    n_channels : int
        Number of channels of the collection.
    window_length : int
        Length ``W`` of the sketch filter.
    itemsize : int
        Size in bytes of one element of the sketch's floating dtype.

    Returns
    -------
    chunk : int
        Number of cases to hash at once, at least 1. A chunk of 1 case is
        returned when a single case does not fit the budget: the pipeline cannot
        split a case any further.
    """
    per_case = (
        n_bits * n_channels * window_length * itemsize
        + _LIVE_SHINGLE_ARRAYS * 8 * n_shingles
    )
    return max(1, _HASH_CHUNK_BYTES // per_case)


def _collection_to_sketch(X, filter_flat, window_length, shift):
    """
    Compute the sliding-window sign sketch of a collection of time series.

    The filter slides over each series with step ``shift``; every position
    contributes one bit, the sign of the inner product between the filter and the
    ``window_length`` values under it. This is a signed random projection of a
    *local* subsequence, so bit agreement is a crude LSH for local shape.

    Each row's bits depend only on that row, and the strided window block is
    materialized for the whole of ``X`` at once, so bounding memory is the
    caller's job: ``SSHIndexANN._hash_collection`` calls this on chunks of at
    most ``_hash_chunk_size`` cases.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Time series collection to sketch.
    filter_flat : np.ndarray of shape (n_channels * window_length,)
        The random filter, flattened in C order from ``(n_channels,
        window_length)``. Its dtype sets the precision of the matrix product.
    window_length : int
        Length ``W`` of the filter.
    shift : int
        Step size ``delta``.

    Returns
    -------
    bits : np.ndarray of shape (n_cases, n_bits), dtype bool
        The sketch of every series, ``True`` where the inner product is >= 0.
    """
    n_cases, n_channels, n_timepoints = X.shape
    n_bits = _n_sketch_bits(n_timepoints, window_length, shift)
    X = X.astype(filter_flat.dtype, copy=False)

    # A view, not a copy: (n_cases, n_channels, n_timepoints - W + 1, W), then
    # every ``shift``-th position.
    windows = sliding_window_view(X, window_length, axis=2)[:, :, ::shift, :]

    # The transpose puts the window axis before the channel axis, so the reshape
    # flattens each window channel-major -- the same C order in which
    # ``filter_flat`` was flattened from (n_channels, window_length). The two
    # orders must agree or a multivariate sketch pairs channels with the wrong
    # filter coefficients.
    block = windows.transpose(0, 2, 1, 3).reshape(-1, n_channels * window_length)
    return (block @ filter_flat).reshape(n_cases, n_bits) >= 0


def _sketch_to_shingle_ids(bits, shingle_size):
    """
    Pack every length-``shingle_size`` window of a sketch into an integer id.

    No sliding window is materialized: bit ``b`` of every shingle is the slice
    ``bits[..., b : b + n_shingles]``, so the ids accumulate in ``shingle_size``
    bit-plane ORs. This mirrors ``_signatures_to_keys`` in
    ``_simhash_index_ann.py`` and keeps peak memory at one uint64 array.

    Parameters
    ----------
    bits : np.ndarray of shape (..., n_bits), dtype bool
        Sketch of one series (1D) or of a collection (2D).
    shingle_size : int
        Shingle length ``n``, at most 64.

    Returns
    -------
    ids : np.ndarray of shape (..., n_bits - shingle_size + 1), dtype uint64
        Integer id of each shingle; bit ``b`` of the shingle contributes
        ``2 ** b``.
    """
    n_shingles = bits.shape[-1] - shingle_size + 1
    ids = np.zeros(bits.shape[:-1] + (n_shingles,), dtype=np.uint64)
    for b in range(shingle_size):
        ids |= bits[..., b : b + n_shingles].astype(np.uint64) << np.uint64(b)
    return ids


def _splitmix64(x):
    """
    Apply the splitmix64 finalizer to a uint64 array.

    All constants are typed ``np.uint64`` to keep the arithmetic explicitly
    modular. Under NEP 50 (numpy >= 2) a Python int operand is weak and would
    stay uint64 anyway, but it raises ``OverflowError`` once the constant exceeds
    ``2 ** 63``, and on numpy 1.x it promoted the whole expression to float64.

    Parameters
    ----------
    x : np.ndarray
        Values to mix; cast to uint64.

    Returns
    -------
    mixed : np.ndarray of dtype uint64
        Well-distributed 64-bit mix of ``x``, same shape as ``x``.
    """
    z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _occurrence_ranks(ids):
    """
    Return how many times each shingle already occurred earlier in its row.

    This materializes the multiset expansion behind the weighted MinHash: a
    shingle with count ``c`` gets ranks ``0 ... c - 1``, so the pairs
    ``(shingle, rank)`` form a plain set whose Jaccard is exactly the weighted
    Jaccard of the shingle counts.

    Parameters
    ----------
    ids : np.ndarray of shape (n_rows, n_shingles), dtype uint64
        Shingle ids, one row per series.

    Returns
    -------
    ranks : np.ndarray of shape (n_rows, n_shingles), dtype uint64
        Occurrence index of each shingle within its own row.
    """
    n_rows, n_shingles = ids.shape
    order = np.argsort(ids, axis=1, kind="stable")
    sorted_ids = np.take_along_axis(ids, order, axis=1)

    new_group = np.empty(sorted_ids.shape, dtype=bool)
    new_group[:, 0] = True
    new_group[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]

    positions = np.broadcast_to(np.arange(n_shingles), sorted_ids.shape)
    # Running max of the positions where a group starts gives, for each sorted
    # element, the start index of the group it belongs to.
    group_start = np.maximum.accumulate(np.where(new_group, positions, 0), axis=1)
    ranks_sorted = (positions - group_start).astype(np.uint64)

    ranks = np.empty_like(ranks_sorted)
    np.put_along_axis(ranks, order, ranks_sorted, axis=1)
    return ranks


def _shingles_to_elements(ids, ranks):
    """
    Map each ``(shingle, occurrence)`` pair to a 64-bit element id.

    Parameters
    ----------
    ids : np.ndarray of dtype uint64
        Shingle ids.
    ranks : np.ndarray of dtype uint64
        Occurrence ranks, aligned with ``ids``.

    Returns
    -------
    elements : np.ndarray of dtype uint64
        Element ids of the expanded multiset, same shape as ``ids``. Distinct
        pairs collide with negligible probability at 64 bits.
    """
    return _splitmix64(ids ^ _splitmix64(ranks))


def _elements_to_minhash(elements, seeds):
    """
    Compute one MinHash value per seed for each row of expanded elements.

    Because the elements are the multiset expansion of the shingle counts, the
    probability that two rows agree on a given seed is exactly the weighted
    Jaccard similarity of their shingle sets.

    Parameters
    ----------
    elements : np.ndarray of shape (n_rows, n_shingles), dtype uint64
        Expanded element ids, as returned by ``_shingles_to_elements``.
    seeds : np.ndarray of shape (n_seeds,), dtype uint64
        One seed per hash function.

    Returns
    -------
    minhashes : np.ndarray of shape (n_rows, n_seeds), dtype uint64
        The minimum mixed element id of each row under each seed.
    """
    minhashes = np.empty((elements.shape[0], seeds.shape[0]), dtype=np.uint64)
    # Loop over seeds rather than broadcasting: peak memory stays at one
    # (n_rows, n_shingles) temporary instead of n_seeds times that.
    for j in range(seeds.shape[0]):
        minhashes[:, j] = _splitmix64(elements + seeds[j]).min(axis=1)
    return minhashes


def _minhash_to_keys(minhashes, n_tables, n_hashes_per_table):
    """
    Fold each table's MinHash values into a single bucket key.

    Parameters
    ----------
    minhashes : np.ndarray of shape (n_rows, n_tables * n_hashes_per_table)
        MinHash values, dtype uint64.
    n_tables : int
        Number of hash tables ``d``.
    n_hashes_per_table : int
        Number of MinHash values ``k`` concatenated into each table key.

    Returns
    -------
    keys : np.ndarray of shape (n_rows, n_tables), dtype uint64
        Bucket key of every row in every table. Two rows share a table's bucket
        only if all ``k`` of that table's MinHash values agree.
    """
    chunks = minhashes.reshape(minhashes.shape[0], n_tables, n_hashes_per_table)
    keys = np.zeros((minhashes.shape[0], n_tables), dtype=np.uint64)
    for i in range(n_hashes_per_table):
        keys = _splitmix64(keys ^ chunks[:, :, i])
    return keys


class SSHIndexANN(BaseWholeSeriesSearch):
    """
    Approximate nearest neighbor search with Sketch, Shingle & Hash (SSH).

    SSH is a data-independent LSH index whose bucket collisions are designed to
    correlate with **DTW** similarity (empirically; see [1]_) rather than with
    cosine similarity. What it provably hashes is the weighted Jaccard similarity
    of shingle multisets, described in step 3 below; DTW is not a metric, so no
    LSH family can be exact for it. Where :class:`SimHashIndexANN` projects the
    whole series and is therefore destroyed by a shift, SSH hashes a
    representation that is invariant to where a pattern occurs:

    1. **Sketch.** One Gaussian filter of length ``window_length`` slides over the
       series with step ``shift``; each position contributes one bit, the sign of
       the inner product. This gives a bit string of
       ``n_sketch_bits_ = (n_timepoints - window_length) // shift + 1`` bits, each
       a 1-bit signed random projection of a short subsequence.
    2. **Shingle.** Every contiguous ``shingle_size``-gram of that bit string is
       counted, giving a *weighted set* of ``n_shingles_`` shingle occurrences.
       Two series sharing a long similar subsequence share many n-grams no matter
       where that subsequence sits, which is where alignment invariance comes
       from.
    3. **Hash.** The set similarity of interest is the weighted Jaccard
       ``sum(min) / sum(max)``. Because the weights are integer counts, expanding
       a shingle of count ``c`` into elements ``(s, 0) ... (s, c - 1)`` makes
       plain MinHash an *exact* LSH for it, so no consistent weighted sampling is
       needed. ``n_tables`` tables are built from those MinHash values.

    A query is sketched, shingled and hashed the same way, then probes one bucket
    per table. The union of those buckets is the candidate set, which is then
    scored with ``distance`` -- the candidates and only the candidates -- and
    returned with true distances, as in the paper's query algorithm. Unlike
    :class:`SimHashIndexANN`, this index does not offer collision count as a
    ranking: too many candidates tie at the top count for it to order anything,
    so the ranking would come down to the index tie-break.

    Note that this method provides **approximate** results: a true neighbor is
    missed if it never shares a bucket with the query. Larger ``n_tables`` raises
    recall and the candidate-set size; larger ``n_hashes_per_table`` makes buckets
    more selective, so candidate sets shrink and queries speed up at the cost of
    recall.

    Parameters
    ----------
    window_length : int
        Length ``W`` of the random filter used for the sketch. Must be at most the
        fitted series length. The paper tunes it per dataset (80 for ECG, 30 for
        Random Walk): too large and a bit merges distinct patterns, too small and
        it only captures noise.
    shift : int
        Step size ``delta`` between two consecutive filter positions. Accuracy
        decreases monotonically with ``shift`` while preprocessing gets cheaper;
        the paper uses 3 and 5.
    shingle_size : int
        Length ``n`` of the bit n-grams, at most 64. A sensitive parameter: the
        paper finds 15 optimal on both its datasets, with accuracy falling off on
        either side.
    n_tables : int, default=20
        Number of hash tables ``d`` (OR-amplification). The paper's value.
    n_hashes_per_table : int, default=1
        Number of MinHash values ``k`` concatenated into each table key
        (AND-amplification). The paper uses one hash per table.
    distance : str or callable, default="dtw"
        Distance used to re-rank the candidate set. A list of valid strings can be
        found in the documentation for
        :func:`aeon.distances.get_distance_function`. The paper's choice is DTW;
        the buckets are built to correlate with it.
    distance_params : dict, default=None
        Dictionary of parameters for ``distance``.
    random_state : int, optional
        Random seed for reproducibility of the filter and the MinHash seeds.
    normalize : bool, default=True
        Whether to z-normalize series before sketching. The fitted collection is
        stored normalized, so re-ranking compares like with like.
    n_jobs : int, default=1
        Number of parallel threads used for the sketch matrix product at fit time
        and for the re-ranking distance computation.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection: z-normalized when ``normalize=True``, raw
        otherwise.
    filter_ : np.ndarray of shape (n_channels, window_length)
        The single Gaussian filter, shared by every series and every table. Table
        independence comes from ``hash_seeds_``, not from redrawing the filter.
    filter_flat_ : np.ndarray of shape (n_channels * window_length,)
        ``filter_`` flattened in C order, in the fitted data's floating precision.
    hash_seeds_ : np.ndarray of shape (n_tables * n_hashes_per_table,), uint64
        One seed per MinHash function.
    tables_ : list of dict
        The ``n_tables`` hash tables, each mapping a uint64 bucket key to an int
        array of the case indices in that bucket.
    n_sketch_bits_ : int
        Length of the sketch bit string.
    n_shingles_ : int
        Number of shingle occurrences, ``n_sketch_bits_ - shingle_size + 1``.
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

    ``window_length``, ``shift`` and ``shingle_size`` have no defaults on purpose:
    the paper's values assume series of 128 to 2048 points, and any fixed default
    is either wrong or degenerate on shorter series.

    See Also
    --------
    SimHashIndexANN : LSH index for cosine similarity over the whole series.
    NaiveSeriesSearch : Exact nearest neighbor search (slower but exact).

    References
    ----------
    .. [1] C. Luo and A. Shrivastava. "SSH (Sketch, Shingle, & Hash) for Indexing
       Massive-Scale Time Series". arXiv:1610.07328, 2016.
    .. [2] A. Z. Broder, S. C. Glassman, M. S. Manasse and G. Zweig. "Syntactic
       clustering of the web". WWW 1997. MinHash for set similarity.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.whole_series import SSHIndexANN
    >>> X_fit = np.random.rand(100, 1, 50)
    >>> query = np.random.rand(1, 50)
    >>> index = SSHIndexANN(window_length=8, shift=2, shingle_size=4)
    >>> index.fit(X_fit)
    SSHIndexANN(shift=2, shingle_size=4, window_length=8)
    >>> indexes, distances = index.predict(query, k=5)
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        window_length,
        shift,
        shingle_size,
        n_tables=20,
        n_hashes_per_table=1,
        distance="dtw",
        distance_params=None,
        random_state=None,
        normalize=True,
        n_jobs=1,
    ):
        self.window_length = window_length
        self.shift = shift
        self.shingle_size = shingle_size
        self.n_tables = n_tables
        self.n_hashes_per_table = n_hashes_per_table
        self.distance = distance
        self.distance_params = distance_params
        self.random_state = random_state
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__()

    def _validate_fit_params(self):
        """Validate the parameters, including against the fitted series length."""
        for name, value in (
            ("window_length", self.window_length),
            ("shift", self.shift),
            ("shingle_size", self.shingle_size),
            ("n_tables", self.n_tables),
            ("n_hashes_per_table", self.n_hashes_per_table),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {value!r}.")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}.")

        if not isinstance(self.normalize, bool):
            raise TypeError(f"normalize must be a bool, got {self.normalize!r}.")

        # Resolve ``distance`` here rather than at the first predict: building the
        # index is the expensive half of this estimator, and a typo that only
        # surfaces once it is paid for wastes exactly what the index is meant to
        # amortise. A callable is returned unchanged.
        try:
            get_distance_function(self.distance)
        except ValueError as error:
            raise ValueError(f"Invalid distance {self.distance!r}: {error}") from error

        if self.window_length > self.n_timepoints_:
            raise ValueError(
                "window_length must be at most the fitted series length "
                f"({self.n_timepoints_}), got {self.window_length}."
            )
        if self.shingle_size > 64:
            raise ValueError(
                "shingle_size must be at most 64 (a shingle is packed into a "
                f"64-bit integer), got {self.shingle_size}."
            )

        n_bits = _n_sketch_bits(self.n_timepoints_, self.window_length, self.shift)
        if n_bits < self.shingle_size:
            raise ValueError(
                f"The sketch of a series of length {self.n_timepoints_} with "
                f"window_length={self.window_length} and shift={self.shift} is "
                f"{n_bits} bits long, which is shorter than "
                f"shingle_size={self.shingle_size}. Decrease shingle_size, "
                "decrease window_length or decrease shift."
            )

    def _fit(self, X, y=None):
        """
        Build the SSH index from X.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Input data to index and search against the query given to predict.
        y : ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._distance_params = self.distance_params or {}
        # Sketch in the caller's floating precision (float64 by default): float32
        # input roughly halves the matmul cost and, since only the sign is kept,
        # leaves the bits unchanged.
        self._input_dtype = (
            X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
        )
        self.n_sketch_bits_ = _n_sketch_bits(
            self.n_timepoints_, self.window_length, self.shift
        )
        self.n_shingles_ = self.n_sketch_bits_ - self.shingle_size + 1

        if self.normalize:
            # Replace the raw collection stored by the base ``fit`` with its
            # z-normalized version: both the sketch and the re-ranking read it,
            # so only one copy is kept.
            X = z_normalise_series_3d(X)
            self.X_ = X

        self._initialize_hash_functions()
        # The sketch is a BLAS matrix product; cap its thread pool to honour
        # n_jobs.
        with threadpool_limits(limits=self._n_jobs, user_api="blas"):
            keys = self._hash_collection(X)
        self.tables_ = _build_hash_tables(keys, self.n_tables)
        return self

    def _initialize_hash_functions(self):
        """Draw the sketch filter and the MinHash seeds."""
        rng = np.random.default_rng(self.random_state)
        self.filter_ = rng.standard_normal(size=(self.n_channels_, self.window_length))
        self.filter_flat_ = self.filter_.reshape(-1).astype(
            self._input_dtype, copy=False
        )
        self.hash_seeds_ = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=self.n_tables * self.n_hashes_per_table,
            dtype=np.uint64,
        )

    def _hash_chunk(self, X):
        """
        Run the full SSH pipeline on a chunk of cases and return its bucket keys.

        Every intermediate array of the pipeline is proportional to the number of
        rows given here, so this must be called on chunks sized by
        ``_hash_chunk_size`` rather than on a whole collection.

        Parameters
        ----------
        X : np.ndarray of shape (n_chunk, n_channels, n_timepoints)
            Cases to hash, already normalized if ``normalize`` is True.

        Returns
        -------
        keys : np.ndarray of shape (n_chunk, n_tables), dtype uint64
            Bucket key of every case of the chunk in every table.
        """
        bits = _collection_to_sketch(
            X, self.filter_flat_, self.window_length, self.shift
        )
        ids = _sketch_to_shingle_ids(bits, self.shingle_size)
        elements = _shingles_to_elements(ids, _occurrence_ranks(ids))
        minhashes = _elements_to_minhash(elements, self.hash_seeds_)
        return _minhash_to_keys(minhashes, self.n_tables, self.n_hashes_per_table)

    def _hash_collection(self, X):
        """
        Hash a collection chunk by chunk and return the bucket keys of every case.

        Every stage of the pipeline is row-independent, so the collection is cut
        into chunks of ``_hash_chunk_size`` cases and the keys of each chunk are
        written into a preallocated output. Peak memory is then set by the chunk
        rather than by ``n_cases``, and the only allocation that grows with the
        collection is ``keys`` itself, at ``8 * n_tables`` bytes per case.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection to hash, already normalized if ``normalize`` is True.

        Returns
        -------
        keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
            Bucket key of every series in every table.
        """
        n_cases = X.shape[0]
        chunk = _hash_chunk_size(
            self.n_sketch_bits_,
            self.n_shingles_,
            self.n_channels_,
            self.window_length,
            np.dtype(self._input_dtype).itemsize,
        )
        keys = np.empty((n_cases, self.n_tables), dtype=np.uint64)
        for start in range(0, n_cases, chunk):
            stop = min(start + chunk, n_cases)
            keys[start:stop] = self._hash_chunk(X[start:stop])
        return keys

    def _hash_series(self, X):
        """
        Run the full SSH pipeline on a single series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Series to hash, already normalized if ``normalize`` is True.

        Returns
        -------
        keys : np.ndarray of shape (n_tables,), dtype uint64
            Bucket key of the series in every table.
        """
        return self._hash_collection(X[np.newaxis])[0]

    def _predict(self, X, k=1, inverse_distance=False):
        """
        Find the k approximate nearest neighbors of a query series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series.
        k : int, default=1
            Number of neighbors to return. ``np.inf`` returns every candidate.
        inverse_distance : bool, default=False
            Not supported by a near-neighbor bucket index. Must be left False;
            passing True raises ``NotImplementedError``.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Indices of the neighbor series, nearest first. ``n_found`` may be
            smaller than ``k`` if too few candidates collide with the query.
        distances : np.ndarray of shape (n_found,)
            True distances under ``distance`` for the returned neighbors.
        """
        if inverse_distance:
            raise NotImplementedError(
                "SSHIndexANN does not support inverse_distance: its buckets "
                "capture near neighbors, not far ones. Use NaiveSeriesSearch "
                "with inverse_distance=True for farthest-neighbor queries."
            )
        self._check_query_length(X)

        if self.normalize:
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

        # The collision count is not used for ranking, only ever as a diagnostic:
        # the candidate set is scored with ``distance`` instead.
        candidates, _ = self._gather_candidates(X)
        return self._rank_candidates(X, candidates, k)

    def _gather_candidates(self, X):
        """
        Tally, per colliding case, in how many tables it shares the query bucket.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series, already normalized if ``normalize`` is True.

        Returns
        -------
        candidates : np.ndarray of shape (n_candidates,), dtype intp
            Distinct case indices colliding with the query in at least one table,
            ascending.
        collisions : np.ndarray of shape (n_candidates,)
            Collision count of each candidate, aligned with ``candidates``.
        """
        keys = self._hash_series(X)
        return _tally_bucket_collisions(self.tables_, keys, self.n_cases_)

    def _rank_candidates(self, X, candidates, k):
        """
        Score the candidate set with ``distance`` and keep the top k.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series, already normalized if ``normalize`` is True.
        candidates : np.ndarray of shape (n_candidates,)
            Distinct candidate case indices, ascending.
        k : int
            Number of neighbors to return.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Top-k candidate indices, nearest first, ties broken by ascending
            index.
        distances : np.ndarray of shape (n_found,)
            True distances aligned with ``indexes``.
        """
        if len(candidates) == 0:
            warnings.warn(
                "No candidates collided with the query in any table; returning "
                "no neighbors. Increase n_tables or decrease n_hashes_per_table.",
                UserWarning,
                stacklevel=3,
            )
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        # Score the candidates only: this is what keeps the query sublinear while
        # still returning true distances.
        distances = pairwise_distance(
            self.X_[candidates],
            X[np.newaxis],
            method=self.distance,
            n_jobs=self._n_jobs,
            **self._distance_params,
        ).reshape(-1)
        order = np.lexsort((candidates, distances))

        n_found = min(k, len(candidates))
        order = order[:n_found]

        if n_found < k:
            warnings.warn(
                f"Only {n_found} candidates collided with the query, fewer than "
                f"the requested k={k}. Increase n_tables or decrease "
                "n_hashes_per_table.",
                UserWarning,
                stacklevel=3,
            )
        return candidates[order], distances[order]

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """
        Return testing parameter settings for the estimator.

        ``window_length``, ``shift`` and ``shingle_size`` have no defaults, so
        these values are what aeon's check harness uses to construct instances.
        They are sized for the 20-timepoint check collection: the sketch is 9
        bits long and holds 7 shingles.
        """
        if parameter_set == "default":
            return [
                {
                    "window_length": 4,
                    "shift": 2,
                    "shingle_size": 3,
                    "n_tables": 4,
                    "n_hashes_per_table": 1,
                },
                {
                    "window_length": 4,
                    "shift": 2,
                    "shingle_size": 3,
                    "n_tables": 4,
                    "n_hashes_per_table": 2,
                },
            ]
        raise NotImplementedError(
            f"The parameter set {parameter_set} is not yet implemented"
        )
