"""Random projection LSH index."""

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.similarity_search.collection._base import BaseCollectionSimilaritySearch
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD, z_normalise_series_3d
from aeon.utils.validation import check_n_jobs


@njit(cache=True)
def _bool_hamming_dist(X, Y):
    """
    Compute a hamming distance on boolean arrays.

    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints)
        A boolean array

    Y : np.ndarray of shape (n_timepoints)
        A boolean array

    Returns
    -------
    d : int
        The hamming distance between X and Y.

    """
    d = np.uint64(0)
    for i in range(X.shape[0]):
        d += X[i] ^ Y[i]
    return d


@njit(cache=True, parallel=True)
def _bool_hamming_dist_matrix(X_bool, collection_bool):
    """
    Compute the distances between X_bool and each boolean array of collection_bool.

    Each array of collection_bool represent the hash value of a bucket in the index.

    Parameters
    ----------
    X_bool : np.ndarray of shape (n_timepoints)
        A 1D boolean array
    collection_bool : np.ndarray of shape (n_cases, n_timepoints)
        A 2D boolean array

    Returns
    -------
    res : np.ndarray of shape (n_cases)
        The distance of X_bool to all buckets in the index

    """
    n_buckets = collection_bool.shape[0]
    res = np.zeros(n_buckets, dtype=np.uint64)
    for i in prange(n_buckets):
        res[i] = _bool_hamming_dist(collection_bool[i], X_bool)
    return res


@njit(cache=True, fastmath=True)
def _nb_flat_dot(X, Y):
    n_channels, n_timepoints = X.shape
    out = 0
    for i in prange(n_channels):
        for j in prange(n_timepoints):
            out += X[i, j] * Y[i, j]
    return out >= 0


@njit(cache=True, parallel=True)
def _collection_to_bool(X, hash_funcs, start_points, length):
    """
    Transform a collection of time series X to their boolean hash representation.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        Time series collection to transform.
    hash_funcs : np.ndarray of shape (n_hash, n_channels, length)
        The random projection vectors used to compute the boolean hash
    start_points : np.ndarray of shape (n_hash)
        The starting index where the random vector should be applied when computing
        the distance to the input series.
    length : int
        Length of the random vectors.

    Returns
    -------
    res : np.ndarray of shape (n_cases, n_hash)
        The boolean representation of all series in X.

    """
    n_hash_funcs = hash_funcs.shape[0]
    n_samples = X.shape[0]
    res = np.empty((n_samples, n_hash_funcs), dtype=np.bool_)
    for j in prange(n_hash_funcs):
        for i in range(n_samples):
            res[i, j] = _nb_flat_dot(
                X[i, :, start_points[j] : start_points[j] + length], hash_funcs[j]
            )
    return res


class RandomProjectionIndexANN(BaseCollectionSimilaritySearch):
    """
    Random Projection Locality Sensitive Hashing index with cosine similarity.

    In this method based on SimHash, we define a hash function as a boolean operation
    such as, given a random vector ``V`` of shape ``(n_channels, L)`` and a time series
    ``X`` of shape ``(n_channels, n_timeponts)`` (with ``L<=n_timepoints``), we compute
    ``X.V > 0`` to obtain the boolean result.
    In the case where ``L<n_timepoints``, each hash function is affected a random
    starting point ``s`` (between ``[0, n_timepoints - L]``) to compute the dot product
    as ``X[:, s:s+L].V``

    Note that this method will not provide exact results, but will perform approximate
    searches. This also ignore any temporal correlation and consider series as
    high dimensional points due to the cosine similarity distance.

    Parameters
    ----------
    n_hash_funcs : int, optional
        Number of random hashing function to use to index series. The default is 128.
    hash_func_coverage : float, optional
        A value in the interval ]0,1] which defines the size L of the random vectors
        relative to the size of the input time series. The default is 0.25.
    use_discrete_vectors: bool, optional,
        Whether to use discrete vectors with values -1 or 1 as random vector. If false,
        the values of the random vectors are drawn uniformly between [-1,1].
    random_state: int, optional
        A random seed to seed the index building. The default is None.
    normalize: bool, optional
        Whether to z-normalize the input the series during fit and predict before
        indexing them.
    n_jobs: int, optional
        Number of parallel threads to use when computing boolean hashes.
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
        X : np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
            Input array to be used to build the index.
        y : optional
            Not used.

        Returns
        -------
        self

        """
        prev_threads = get_num_threads()
        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        rng = np.random.default_rng(self.random_state)
        if self.normalize:
            X = z_normalise_series_3d(X)
        self.n_timepoints_ = X.shape[2]
        self.window_length_ = max(1, int(self.n_timepoints_ * self.hash_func_coverage))

        if self.use_discrete_vectors:
            self.hash_funcs_ = rng.choice(
                [-1, 1], size=(self.n_hash_funcs, self.n_channels_, self.window_length_)
            )
        else:
            self.hash_funcs_ = rng.uniform(
                low=-1,
                high=1.0,
                size=(self.n_hash_funcs, self.n_channels_, self.window_length_),
            )
        self.start_points_ = rng.choice(
            self.n_timepoints_ - self.window_length_ + 1,
            size=self.n_hash_funcs,
            replace=True,
        )
        X_bools = self._collection_to_hashes(X)
        self.index_ = {}
        self._raw_index_bool_arrays = np.unique(X_bools, axis=0)
        for i in range(len(X_bools)):
            key = X_bools[i].tostring()
            if key in self.index_:
                self.index_[key].append(i)
            else:
                self.index_[key] = [i]
        set_num_threads(prev_threads)
        return self

    def _predict(
        self,
        X,
        k=1,
        inverse_distance=False,
    ):
        """
        Find approximate nearest neighbors of a collection in the index.

        Parameters
        ----------
        X : np.ndarray, shape = (n_cases, n_channels, n_tiempoints)
            Collections of series for which we want to find neighbors.
        k : int, optional
            Number of neighbors to return for each series. The default is 1.
        inverse_distance : bool, optional
            Whether to inverse the computed distance, meaning that the method will
            return the k most dissimilar neighbors instead of the k most similar.

        Returns
        -------
        top_k : np.ndarray, shape = (n_cases, k)
            Indexes of k series in the index that are similar to X.
        top_k_dist : np.ndarray, shape = (n_cases, k)
            Distance of k series in the index to X. The distance
            is the hamming distance between the result of each hash function.
        """
        if X[0].shape[1] != self.n_timepoints_:
            raise ValueError(
                "Expected series of the same length as the series given in fit, but got"
                f"{X[0].shape[1]} instead of {self.n_timepoints_}"
            )

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)
        if self.normalize:
            X = z_normalise_series_3d(X)

        X_bools = self._collection_to_hashes(X)
        top_k = []
        top_k_dist = []

        for i in range(len(X_bools)):
            idx, dists = self._extract_neighors_one_series(
                X_bools[i],
                k=k,
                inverse_distance=inverse_distance,
            )
            top_k.append(idx)
            top_k_dist.append(dists)

        set_num_threads(prev_threads)
        return top_k, top_k_dist

    def _extract_neighors_one_series(
        self,
        X_bool,
        k=1,
        inverse_distance=False,
    ):
        key = X_bool.tostring()
        top_k = np.zeros(k, dtype=int)
        top_k_dist = np.zeros(k, dtype=float)
        remove_X_hash = False
        if not inverse_distance and key in self.index_:
            current_k = min(len(self.index_[key]), k)
            top_k[:current_k] = self.index_[key][:current_k]
            remove_X_hash = True
        else:
            current_k = 0

        # Case where we want to find more neighbors in buckets with similar hash
        if current_k < k:
            dists = _bool_hamming_dist_matrix(X_bool, self._raw_index_bool_arrays)

            if inverse_distance:
                dists = 1 / (dists + AEON_NUMBA_STD_THRESHOLD)
            if remove_X_hash:
                dists[np.where(self._raw_index_bool_arrays == key)[0]] = np.iinfo(
                    dists.dtype
                ).max
            # Get top k index of keys
            ids = np.argpartition(dists, kth=k)[:k]
            # and reorder them
            ids = ids[np.argsort(dists[ids])]

            _i_bucket = 0
            while current_k < k:
                key_index = self._raw_index_bool_arrays[ids[_i_bucket]].tostring()
                candidates = self.index_[key_index]
                # Can do exact search by computing distances here
                if len(candidates) > k - current_k:
                    candidates = candidates[: k - current_k]
                top_k[current_k : current_k + len(candidates)] = candidates
                top_k_dist[current_k : current_k + len(candidates)] = dists[
                    ids[_i_bucket]
                ]
                current_k += len(candidates)
                _i_bucket += 1

        return top_k[:current_k], top_k_dist[:current_k]

    def _collection_to_hashes(self, X):
        return _collection_to_bool(
            X, self.hash_funcs_, self.start_points_, self.window_length_
        )
