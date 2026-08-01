"""SSH (Sketch, Shingle & Hash) index."""

__maintainer__ = ["baraline"]
__all__ = ["SSHIndexANN"]

import warnings

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

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

# Number of cases one parallel task hashes. Each task allocates the scratch
# buffers of ``_hash_collection`` once and reuses them across its cases, so this
# amortises the allocation; keeping it small keeps the tasks balanced. The kernel
# lowers it further when the collection is too small to give every thread a task.
_HASH_BLOCK = 16

# splitmix64 constants. All are typed ``np.uint64`` so the arithmetic is
# explicitly modular: numba infers uint64 from these, whereas a Python int
# literal would make the expression float64.
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MIX2 = np.uint64(0x94D049BB133111EB)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)
_UINT64_ONE = np.uint64(1)
_UINT64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)


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


@njit(cache=True, inline="always")
def _splitmix64(x):
    """
    Apply the splitmix64 finalizer to a 64-bit value.

    Parameters
    ----------
    x : np.uint64
        Value to mix.

    Returns
    -------
    mixed : np.uint64
        Well-distributed 64-bit mix of ``x``.
    """
    z = x + _SPLITMIX_GAMMA
    z = (z ^ (z >> _SHIFT_30)) * _SPLITMIX_MIX1
    z = (z ^ (z >> _SHIFT_27)) * _SPLITMIX_MIX2
    return z ^ (z >> _SHIFT_31)


@njit(cache=True, fastmath=True, inline="always")
def _series_to_sketch(x, filter_, shift, bits):
    """
    Write the sliding-window sign sketch of one series into ``bits``.

    The filter slides over the series with step ``shift``; every position
    contributes one bit, the sign of the inner product between the filter and the
    ``window_length`` values under it. This is a signed random projection of a
    *local* subsequence, so bit agreement is a crude LSH for local shape.

    The window block is never materialized: each bit reads its window straight
    out of ``x``, which stays in cache across the whole sketch. That is what makes
    the sketch cheap despite each point being read up to ``W / shift`` times.

    Parameters
    ----------
    x : np.ndarray of shape (n_channels, n_timepoints)
        Series to sketch.
    filter_ : np.ndarray of shape (n_channels, window_length)
        The random filter. Channel ``c`` of a window is paired with row ``c`` of
        the filter, so a multivariate sketch mixes the channels of one window.
    shift : int
        Step size ``delta``.
    bits : np.ndarray of shape (n_bits,), dtype bool
        Output buffer, ``True`` where the inner product is >= 0.
    """
    n_channels, window_length = filter_.shape
    for j in range(bits.shape[0]):
        # Accumulate in float64 whatever the input precision: the sketch is a
        # sign test, so the bits must not depend on the dtype the caller happens
        # to pass. It costs nothing -- the loop is latency-bound, not
        # bandwidth-bound, and float32 input times the same either way.
        inner = 0.0
        offset = j * shift
        for c in range(n_channels):
            for w in range(window_length):
                inner += x[c, offset + w] * filter_[c, w]
        bits[j] = inner >= 0


@njit(cache=True, inline="always")
def _sketch_to_minhash(bits, shingle_size, seeds, minhashes, ids, counts, stamps, tag):
    """
    Shingle a sketch and reduce it to one MinHash value per seed.

    Steps 2 and 3 of SSH share a single pass over the shingle positions, because
    every intermediate is consumed where it is produced: a shingle id feeds its
    own occurrence rank, the pair feeds one element id, and the element updates
    the running minima. Materializing them instead -- an ``(n_shingles,)`` array
    per intermediate -- is what the numpy implementation used to do, and it is
    slower even though the arrays are small enough to stay in cache.

    Two loop-level identities keep the pass linear:

    - **Rolling shingle id.** Consecutive shingles overlap in all but one bit, so
      the id of shingle ``j + 1`` is the id of shingle ``j`` shifted down by one
      with the newly entered bit placed on top. That is O(1) per shingle instead
      of the O(``shingle_size``) repacking of every window.
    - **Occurrence ranks by open addressing.** The rank of a shingle is how many
      times it already occurred in this series, which a hash table counts in one
      pass; the numpy implementation needed a sort of the whole row. The table is
      shared across the cases of a parallel task and never cleared: an entry
      counts for the current case only if ``stamps`` marks it with ``tag``, so a
      stale entry from the previous case reads as empty.

    Ranking occurrences this way is the multiset expansion behind the weighted
    MinHash: a shingle with count ``c`` contributes elements
    ``(s, 0) ... (s, c - 1)``, so the pairs form a plain set whose Jaccard is
    exactly the weighted Jaccard of the shingle counts, and plain MinHash over
    them is an exact LSH for it.

    Parameters
    ----------
    bits : np.ndarray of shape (n_bits,), dtype bool
        Sketch of one series.
    shingle_size : int
        Shingle length ``n``, at most 64.
    seeds : np.ndarray of shape (n_seeds,), dtype uint64
        One seed per hash function.
    minhashes : np.ndarray of shape (n_seeds,), dtype uint64
        Output buffer: the minimum mixed element id under each seed.
    ids : np.ndarray of shape (table_size,), dtype uint64
        Scratch: keys of the open-addressing table. ``table_size`` must be a
        power of two of at least ``2 * n_shingles``, so the table never fills.
    counts : np.ndarray of shape (table_size,), dtype int64
        Scratch: occurrence count of each live table entry.
    stamps : np.ndarray of shape (table_size,), dtype int64
        Scratch: which case each table entry belongs to. Must be filled with a
        value no case ever uses as ``tag`` (-1) before the first call.
    tag : int
        Identifier of the current case, unique within the calls that share these
        scratch buffers.
    """
    n_shingles = bits.shape[0] - shingle_size + 1
    n_seeds = seeds.shape[0]
    mask = np.uint64(ids.shape[0] - 1)
    top_bit = np.uint64(shingle_size - 1)

    for s in range(n_seeds):
        minhashes[s] = _UINT64_MAX

    shingle = np.uint64(0)
    for b in range(shingle_size):
        if bits[b]:
            shingle |= _UINT64_ONE << np.uint64(b)

    for j in range(n_shingles):
        if j > 0:
            shingle >>= _UINT64_ONE
            if bits[j + shingle_size - 1]:
                shingle |= _UINT64_ONE << top_bit

        slot = _splitmix64(shingle) & mask
        while stamps[slot] == tag and ids[slot] != shingle:
            slot = (slot + _UINT64_ONE) & mask
        if stamps[slot] != tag:
            stamps[slot] = tag
            ids[slot] = shingle
            counts[slot] = 0
        rank = np.uint64(counts[slot])
        counts[slot] += 1

        # Element id of the (shingle, occurrence) pair. Mixing the rank before
        # the xor stops pairs that differ by a swap of the two from colliding;
        # at 64 bits distinct pairs collide with negligible probability.
        element = _splitmix64(shingle ^ _splitmix64(rank))
        for s in range(n_seeds):
            value = _splitmix64(element + seeds[s])
            if value < minhashes[s]:
                minhashes[s] = value


@njit(cache=True, inline="always")
def _minhash_to_keys(minhashes, n_hashes_per_table, keys):
    """
    Fold each table's MinHash values into a single bucket key.

    Parameters
    ----------
    minhashes : np.ndarray of shape (n_tables * n_hashes_per_table,), uint64
        MinHash values of one series, table-major.
    n_hashes_per_table : int
        Number of MinHash values ``k`` concatenated into each table key.
    keys : np.ndarray of shape (n_tables,), dtype uint64
        Output buffer: bucket key of the series in every table. Two series share
        a table's bucket only if all ``k`` of that table's MinHash values agree.
    """
    for t in range(keys.shape[0]):
        key = np.uint64(0)
        base = t * n_hashes_per_table
        for m in range(n_hashes_per_table):
            key = _splitmix64(key ^ minhashes[base + m])
        keys[t] = key


@njit(cache=True, parallel=True)
def _hash_collection_kernel(
    X, filter_, shift, shingle_size, seeds, n_tables, n_per_table, block
):
    """
    Run the whole SSH pipeline on a collection and return its bucket keys.

    A case is sketched, shingled and hashed without ever leaving the scratch
    buffers of the task that owns it, so the only array that grows with the
    collection is the output. Peak memory is a few kilobytes per thread instead
    of the ``(n_cases, n_shingles)`` intermediates of a staged implementation --
    which is why this needs none of the chunking that one required.

    Cases are handed out in blocks rather than one by one so that the scratch
    buffers, in particular the occurrence table, are allocated once per task and
    reused across its cases. ``block`` is passed in rather than derived from
    ``get_num_threads()`` here: that function is backed by a ctypes pointer,
    which numba counts as a dynamic global and which would silently drop this
    kernel's ``cache=True``, making every fresh process pay the compilation
    again.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Collection to hash, already normalized if ``normalize`` is True.
    filter_ : np.ndarray of shape (n_channels, window_length)
        The random filter.
    shift : int
        Step size ``delta``.
    shingle_size : int
        Shingle length ``n``.
    seeds : np.ndarray of shape (n_tables * n_hashes_per_table,), uint64
        One seed per MinHash function.
    n_tables : int
        Number of hash tables ``d``.
    n_per_table : int
        Number of MinHash values ``k`` folded into each table key.
    block : int
        Number of cases per parallel task, at least 1.

    Returns
    -------
    keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
        Bucket key of every case in every table.
    """
    n_cases, _, n_timepoints = X.shape
    n_bits = (n_timepoints - filter_.shape[1]) // shift + 1
    n_shingles = n_bits - shingle_size + 1

    # Load factor at most 1/2, so linear probing stays short and the table can
    # never fill up.
    table_size = 1
    while table_size < 2 * n_shingles:
        table_size *= 2

    n_blocks = (n_cases + block - 1) // block
    keys = np.empty((n_cases, n_tables), dtype=np.uint64)
    for b in prange(n_blocks):
        bits = np.empty(n_bits, dtype=np.bool_)
        minhashes = np.empty(seeds.shape[0], dtype=np.uint64)
        ids = np.empty(table_size, dtype=np.uint64)
        counts = np.empty(table_size, dtype=np.int64)
        stamps = np.full(table_size, -1, dtype=np.int64)
        for i in range(b * block, min((b + 1) * block, n_cases)):
            _series_to_sketch(X[i], filter_, shift, bits)
            _sketch_to_minhash(
                bits, shingle_size, seeds, minhashes, ids, counts, stamps, i
            )
            _minhash_to_keys(minhashes, n_per_table, keys[i])
    return keys


def _hash_collection(X, filter_, shift, shingle_size, seeds, n_tables, n_per_table):
    """
    Hash a collection with ``_hash_collection_kernel``, sizing its tasks.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Collection to hash, already normalized if ``normalize`` is True.
    filter_ : np.ndarray of shape (n_channels, window_length)
        The random filter.
    shift : int
        Step size ``delta``.
    shingle_size : int
        Shingle length ``n``.
    seeds : np.ndarray of shape (n_tables * n_hashes_per_table,), uint64
        One seed per MinHash function.
    n_tables : int
        Number of hash tables ``d``.
    n_per_table : int
        Number of MinHash values ``k`` folded into each table key.

    Returns
    -------
    keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
        Bucket key of every case in every table.
    """
    n_threads = get_num_threads()
    # ``_HASH_BLOCK`` is an upper bound, not a target: a collection too small to
    # give every thread a block is split finer instead.
    block = min(_HASH_BLOCK, max(1, -(-X.shape[0] // n_threads)))
    return _hash_collection_kernel(
        np.ascontiguousarray(X),
        filter_,
        shift,
        shingle_size,
        seeds,
        n_tables,
        n_per_table,
        block,
    )


class SSHIndexANN(BaseWholeSeriesSearch):
    """
    Approximate nearest neighbor search with Sketch, Shingle & Hash (SSH).

    SSH is a data-independent LSH index whose bucket collisions are designed to
    correlate with **DTW** similarity (empirically; see [1]_) rather than with
    cosine similarity. What it truly hash is the weighted Jaccard similarity
    of shingle multisets. (DTW is not a metric, so no LSH family can be exact
    for it). The main advantage of SSH is that it hashes a representation that
    is invariant to where a pattern occurs, doing so in 3 steps:

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

    This method provides **approximate** results: a true neighbor is
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
        stored normalized, so re-ranking compares like with like. Read at fit time:
        ``X_`` is stored on the scale it selects, so setting it on a fitted
        estimator has no effect until that estimator is refitted.
    n_jobs : int, default=1
        Number of parallel threads used to hash the collection at fit time and
        for the re-ranking distance computation. Cases are hashed independently,
        so fit time scales close to linearly with this.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection: z-normalized when ``normalize=True``, raw
        otherwise.
    filter_ : np.ndarray of shape (n_channels, window_length)
        The single Gaussian filter, shared by every series and every table. Table
        independence comes from ``hash_seeds_``, not from redrawing the filter.
    hash_seeds_ : np.ndarray of shape (n_tables * n_hashes_per_table,), uint64
        One seed per MinHash function.
    tables_ : HashTables
        The ``n_tables`` hash tables, holding the case indices of every bucket
        in a flat compressed layout. ``_bucket_dicts`` materializes them as one
        ``{bucket key: case indices}`` dict per table.
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
        self._normalize = self.normalize
        self.n_sketch_bits_ = _n_sketch_bits(
            self.n_timepoints_, self.window_length, self.shift
        )
        self.n_shingles_ = self.n_sketch_bits_ - self.shingle_size + 1

        if self._normalize:
            # Replace the raw collection stored by the base ``fit`` with its
            # z-normalized version: both the sketch and the re-ranking read it,
            # so only one copy is kept.
            X = z_normalise_series_3d(X)
            self.X_ = X

        self._initialize_hash_functions()
        keys = self._hash_collection(X)
        self.tables_ = _build_hash_tables(keys, self.n_tables)
        return self

    def _initialize_hash_functions(self):
        """Draw the sketch filter and the MinHash seeds."""
        rng = np.random.default_rng(self.random_state)
        self.filter_ = rng.standard_normal(size=(self.n_channels_, self.window_length))
        self.hash_seeds_ = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=self.n_tables * self.n_hashes_per_table,
            dtype=np.uint64,
        )

    def _hash_collection(self, X):
        """
        Run the SSH pipeline on a collection and return the bucket keys.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection to hash, already normalized if ``normalize`` is True.

        Returns
        -------
        keys : np.ndarray of shape (n_cases, n_tables), dtype uint64
            Bucket key of every series in every table.
        """
        previous_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        try:
            return _hash_collection(
                X,
                self.filter_,
                self.shift,
                self.shingle_size,
                self.hash_seeds_,
                self.n_tables,
                self.n_hashes_per_table,
            )
        finally:
            set_num_threads(previous_threads)

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
