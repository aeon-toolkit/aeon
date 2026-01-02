"""Random projection LSH index."""

import warnings

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
from aeon.utils.numba.general import (
    AEON_NUMBA_STD_THRESHOLD,
    z_normalise_series_2d,
    z_normalise_series_3d,
)
from aeon.utils.validation import check_n_jobs


@njit(cache=True)
def _bool_hamming_dist(X, Y):
    """
    Compute Hamming distance between two boolean arrays.

    Parameters
    ----------
    X : np.ndarray of shape (n_hash,)
        A boolean array.
    Y : np.ndarray of shape (n_hash,)
        A boolean array.

    Returns
    -------
    d : uint64
        The Hamming distance between X and Y.
    """
    d = np.uint64(0)
    for i in range(X.shape[0]):
        d += X[i] ^ Y[i]
    return d


@njit(cache=True, parallel=True)
def _bool_hamming_dist_matrix(X_bool, collection_bool):
    """
    Compute Hamming distances between a query hash and all bucket hashes.

    Parameters
    ----------
    X_bool : np.ndarray of shape (n_hash,)
        Query boolean hash array.
    collection_bool : np.ndarray of shape (n_buckets, n_hash)
        Boolean hash arrays for all buckets.

    Returns
    -------
    res : np.ndarray of shape (n_buckets,)
        Hamming distance from query to each bucket.
    """
    n_buckets = collection_bool.shape[0]
    res = np.zeros(n_buckets, dtype=np.uint64)
    for i in prange(n_buckets):
        res[i] = _bool_hamming_dist(collection_bool[i], X_bool)
    return res


@njit(cache=True, fastmath=True)
def _dot_product_sign(X, Y):
    """
    Compute sign of dot product between two 2D arrays.

    Parameters
    ----------
    X : np.ndarray of shape (n_channels, length)
        First array (time series segment).
    Y : np.ndarray of shape (n_channels, length)
        Second array (random projection vector).

    Returns
    -------
    bool
        True if dot product >= 0, False otherwise.
    """
    n_channels, n_timepoints = X.shape
    out = 0.0
    for i in range(n_channels):
        for j in range(n_timepoints):
            out += X[i, j] * Y[i, j]
    return out >= 0


@njit(cache=True, parallel=True)
def _series_to_bool(X, hash_funcs, start_points, length):
    """
    Compute boolean hash for a single time series.

    Parameters
    ----------
    X : np.ndarray of shape (n_channels, n_timepoints)
        Time series to hash.
    hash_funcs : np.ndarray of shape (n_hash, n_channels, length)
        Random projection vectors.
    start_points : np.ndarray of shape (n_hash,)
        Starting indices for each hash function.
    length : int
        Length of random projection vectors.

    Returns
    -------
    res : np.ndarray of shape (n_hash,)
        Boolean hash representation of the time series.
    """
    n_hash_funcs = hash_funcs.shape[0]
    res = np.zeros(n_hash_funcs, dtype=np.bool_)
    for j in prange(n_hash_funcs):
        start = start_points[j]
        res[j] = _dot_product_sign(X[:, start : start + length], hash_funcs[j])
    return res


@njit(cache=True, parallel=True)
def _collection_to_bool(X, hash_funcs, start_points, length):
    """
    Compute boolean hashes for a collection of time series.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Time series collection to hash.
    hash_funcs : np.ndarray of shape (n_hash, n_channels, length)
        Random projection vectors.
    start_points : np.ndarray of shape (n_hash,)
        Starting indices for each hash function.
    length : int
        Length of random projection vectors.

    Returns
    -------
    res : np.ndarray of shape (n_cases, n_hash)
        Boolean hash representation of all time series.
    """
    n_hash_funcs = hash_funcs.shape[0]
    n_samples = X.shape[0]
    res = np.zeros((n_samples, n_hash_funcs), dtype=np.bool_)
    for i in prange(n_samples):
        res[i, :] = _series_to_bool(X[i], hash_funcs, start_points, length)
    return res


class LSHIndex(BaseWholeSeriesSearch):
    """
    Approximate nearest neighbor search using Locality Sensitive Hashing.

    This method uses Random Projection LSH with cosine similarity (SimHash) for
    fast approximate nearest neighbor search. Each series is hashed to a binary
    signature using random projection vectors, and similar series are likely to
    have similar hash signatures.

    The hash function computes ``sign(X.V)`` where ``V`` is a random vector of
    shape ``(n_channels, L)`` and ``X`` is a time series of shape
    ``(n_channels, n_timepoints)``. When ``L < n_timepoints``, a random starting
    point ``s`` is used to compute ``sign(X[:, s:s+L].V)``.

    Note that this method provides **approximate** results, trading accuracy for
    speed. It ignores temporal correlation and treats series as high-dimensional
    points using cosine similarity.

    Parameters
    ----------
    n_hash_funcs : int, default=128
        Number of random hash functions to use. More functions increase accuracy
        but also memory usage and computation time.
    hash_func_coverage : float, default=0.25
        A value in (0, 1] defining the size ``L`` of random vectors relative to
        the time series length: ``L = n_timepoints * hash_func_coverage``.
    use_discrete_vectors : bool, default=True
        If True, random vectors have values {-1, 1}. If False, values are drawn
        uniformly from [-1, 1].
    random_state : int, optional
        Random seed for reproducibility of hash function generation.
    normalize : bool, default=True
        Whether to z-normalize series before indexing. Recommended for
        scale-independent matching.
    n_jobs : int, default=1
        Number of parallel threads for computing hashes.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The fitted collection of time series.
    index_ : dict
        Hash table mapping binary signatures to lists of case indices.
    hash_funcs_ : np.ndarray
        The random projection vectors used for hashing.
    n_cases_ : int
        Number of time series in the fitted collection.
    n_channels_ : int
        Number of channels in the fitted time series.
    n_timepoints_ : int
        Number of timepoints in each fitted time series.

    See Also
    --------
    BruteForce : Exact nearest neighbor search (slower but exact).

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.similarity_search.whole_series import LSHIndex
    >>> # Create a collection of 100 univariate time series
    >>> X_fit = np.random.rand(100, 1, 50)
    >>> # Create a query series
    >>> query = np.random.rand(1, 50)
    >>> # Initialize and fit the index
    >>> index = LSHIndex(n_hash_funcs=128, normalize=True)
    >>> index.fit(X_fit)  # doctest: +SKIP
    >>> # Find approximate nearest neighbors
    >>> indexes, distances = index.predict(query, k=5)  # doctest: +SKIP
    >>> # indexes is shape (5,) with case indices
    >>> # distances is shape (5,) with Hamming distances (lower is more similar)
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        n_hash_funcs=128,
        hash_func_coverage=0.25,
        use_discrete_vectors=True,
        random_state=None,
        normalize=True,
        n_jobs=1,
    ):
        self.n_hash_funcs = n_hash_funcs
        self.hash_func_coverage = hash_func_coverage
        self.use_discrete_vectors = use_discrete_vectors
        self.random_state = random_state
        self.normalize = normalize
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y=None):
        """
        Build the index based on the X.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            Input data to store and use as database against the query given when calling
            predict.
        y : ignored, exists for API consistency reasons.

        Returns
        -------
        self : a fitted instance of the estimator
        """
        prev_threads = get_num_threads()
        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        if self.normalize:
            X = z_normalise_series_3d(X)

        self._initialize_hash_functions(X.shape[2])
        self._build_index(X)

        set_num_threads(prev_threads)
        return self

    def _initialize_hash_functions(self, n_timepoints):
        """
        Initialize random projection vectors and their starting points.

        Parameters
        ----------
        n_timepoints : int
            Length of input time series.
        """
        rng = np.random.default_rng(self.random_state)

        self.n_timepoints_ = n_timepoints
        self.window_length_ = max(1, int(n_timepoints * self.hash_func_coverage))

        shape = (self.n_hash_funcs, self.n_channels_, self.window_length_)
        if self.use_discrete_vectors:
            self.hash_funcs_ = rng.choice([-1, 1], size=shape)
        else:
            self.hash_funcs_ = rng.uniform(low=-1, high=1.0, size=shape)

        n_possible_starts = n_timepoints - self.window_length_ + 1
        self.start_points_ = rng.choice(
            n_possible_starts, size=self.n_hash_funcs, replace=True
        )

    def _build_index(self, X):
        """
        Build the hash table index from time series collection.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Time series collection to index.
        """
        X_bools = _collection_to_bool(
            X, self.hash_funcs_, self.start_points_, self.window_length_
        )

        self._raw_index_bool_arrays = np.unique(X_bools, axis=0)
        self.index_ = {}

        for i in range(len(X_bools)):
            key = X_bools[i].tobytes()
            if key in self.index_:
                self.index_[key].append(i)
            else:
                self.index_[key] = [i]

    def _predict(self, X, k=1, inverse_distance=False):
        """
        Find approximate nearest neighbors for a query series.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Query series.
        k : int, optional
            Number of neighbors to return. Default is 1.
        inverse_distance : bool, optional
            If True, return k most dissimilar series. Default is False.

        Returns
        -------
        indexes : np.ndarray of shape (n_found,)
            Indices of neighbor series in the database.
        distances : np.ndarray of shape (n_found,)
            Hamming distances to the neighbors.
        """
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        if self.normalize:
            X = z_normalise_series_2d(X)

        X_bool = _series_to_bool(
            X, self.hash_funcs_, self.start_points_, self.window_length_
        )
        indexes, distances = self._find_neighbors(X_bool, k, inverse_distance)

        set_num_threads(prev_threads)
        return indexes, distances

    def _find_neighbors(self, X_bool, k, inverse_distance):
        """
        Find k nearest (or farthest) neighbors for a query hash.

        Parameters
        ----------
        X_bool : np.ndarray of shape (n_hash,)
            Boolean hash of the query series.
        k : int
            Number of neighbors to find.
        inverse_distance : bool
            If True, find most dissimilar instead of most similar.

        Returns
        -------
        top_k : np.ndarray of shape (n_found,)
            Indices of neighbor series.
        top_k_dist : np.ndarray of shape (n_found,)
            Distances to the neighbors.
        """
        # Cap k at number of indexed cases
        if k > self.n_cases_:
            warnings.warn(
                f"k={k} is larger than the number of indexed cases ({self.n_cases_}). "
                f"Returning at most {self.n_cases_} neighbors.",
                UserWarning,
                stacklevel=3,
            )
            k = self.n_cases_

        top_k = np.zeros(k, dtype=int)
        top_k_dist = np.zeros(k, dtype=float)
        current_k = 0

        # First, check if query hash exists in index (for similar search)
        key = X_bool.tobytes()
        exclude_key = False
        if not inverse_distance and key in self.index_:
            candidates = self.index_[key]
            n_from_bucket = min(len(candidates), k)
            top_k[:n_from_bucket] = candidates[:n_from_bucket]
            current_k = n_from_bucket
            exclude_key = True

        # Find more neighbors from nearby buckets if needed
        if current_k < k:
            current_k = self._search_nearby_buckets(
                X_bool, k, inverse_distance, exclude_key, top_k, top_k_dist, current_k
            )

        return top_k[:current_k], top_k_dist[:current_k]

    def _search_nearby_buckets(
        self, X_bool, k, inverse_distance, exclude_key, top_k, top_k_dist, current_k
    ):
        """
        Search nearby buckets to find additional neighbors.

        Parameters
        ----------
        X_bool : np.ndarray of shape (n_hash,)
            Boolean hash of query series.
        k : int
            Total number of neighbors needed.
        inverse_distance : bool
            If True, find most dissimilar.
        exclude_key : bool
            If True, exclude the query's own bucket from results.
        top_k : np.ndarray of shape (k,)
            Array to store neighbor indices (modified in place).
        top_k_dist : np.ndarray of shape (k,)
            Array to store distances (modified in place).
        current_k : int
            Number of neighbors already found.

        Returns
        -------
        current_k : int
            Updated count of neighbors found.
        """
        dists = _bool_hamming_dist_matrix(X_bool, self._raw_index_bool_arrays)
        n_buckets = len(dists)

        # Transform distances if searching for dissimilar series
        if inverse_distance:
            dists = 1.0 / (dists.astype(np.float64) + AEON_NUMBA_STD_THRESHOLD)

        # Exclude query's bucket if needed
        if exclude_key:
            self._exclude_matching_bucket(X_bool, dists, inverse_distance)

        # Get indices of closest buckets
        bucket_order = self._get_sorted_bucket_indices(dists, k, n_buckets)

        # Collect candidates from buckets
        for bucket_idx in bucket_order:
            if current_k >= k:
                break

            bucket_key = self._raw_index_bool_arrays[bucket_idx].tobytes()
            candidates = self.index_[bucket_key]
            n_to_add = min(len(candidates), k - current_k)

            top_k[current_k : current_k + n_to_add] = candidates[:n_to_add]
            top_k_dist[current_k : current_k + n_to_add] = dists[bucket_idx]
            current_k += n_to_add

        return current_k

    def _exclude_matching_bucket(self, X_bool, dists, inverse_distance):
        """
        Set distance of query's own bucket to worst so it doesn't match itself.

        Parameters
        ----------
        X_bool : np.ndarray of shape (n_hash,)
            Boolean hash of query series.
        dists : np.ndarray of shape (n_buckets,)
            Distance array (modified in place).
        inverse_distance : bool
            Whether distances are inverted.
        """
        match_idx = np.where((self._raw_index_bool_arrays == X_bool).all(axis=1))[0]
        if len(match_idx) > 0:
            if inverse_distance:
                dists[match_idx[0]] = 0.0  # Worst for inverted (want high values)
            else:
                dists[match_idx[0]] = np.iinfo(np.uint64).max  # Worst for normal

    def _get_sorted_bucket_indices(self, dists, k, n_buckets):
        """
        Get indices of buckets sorted by distance.

        Parameters
        ----------
        dists : np.ndarray of shape (n_buckets,)
            Distances to each bucket.
        k : int
            Number of neighbors needed.
        n_buckets : int
            Total number of buckets.

        Returns
        -------
        sorted_indices : np.ndarray
            Bucket indices sorted by distance (ascending).
        """
        n_to_consider = min(k, n_buckets)
        kth = max(0, n_to_consider - 1)
        top_indices = np.argpartition(dists, kth=kth)[:n_to_consider]
        return top_indices[np.argsort(dists[top_indices])]
