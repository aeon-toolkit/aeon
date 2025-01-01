"""Random projection LSH index."""

import numpy as np
from numba import njit, prange

TPB = 16


@njit(cache=True)
def _hamming_dist(X, Y):
    d = 0
    for i in prange(X.shape[0]):
        d += X[i] ^ Y[i]
    return d


@njit(cache=True, parallel=True)
def _hamming_dist_matrix(bool_hashes_value_list, bool_hashes):
    n_hashes = bool_hashes.shape[0]
    res = np.zeros((n_hashes, bool_hashes_value_list.shape[0]), dtype=np.int64)
    for i in prange(n_hashes):
        for j in prange(bool_hashes_value_list.shape[0]):
            res[i, j] = _hamming_dist(bool_hashes_value_list[j], bool_hashes[i])
    return res


@njit(cache=True, fastmath=True, parallel=True)
def _series_to_bool(X, hash_funcs, start_points, length):
    n_hash_funcs = hash_funcs.shape[0]
    res = np.empty(n_hash_funcs, dtype=np.bool_)
    for j in prange(n_hash_funcs):
        res[j] = (
            np.dot(X[start_points[j] : start_points[j] + length], hash_funcs[j]) >= 0
        )
    return res


@njit(cache=True, fastmath=True, parallel=True)
def _collection_to_bool(X, hash_funcs, start_points, length):
    n_hash_funcs = hash_funcs.shape[0]
    n_samples = X.shape[0]
    res = np.empty((n_samples, n_hash_funcs), dtype=np.bool_)
    for i in prange(n_samples):
        for j in prange(n_hash_funcs):
            res[i, j] = (
                np.dot(X[i, start_points[j] : start_points[j] + length], hash_funcs[j])
                >= 0
            )
    return res


class LSH:
    """
    .

    Parameters
    ----------
    n_vectors : TYPE
        DESCRIPTION.
    custom_table : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    def __init__(self, n_hash_funcs=128, window_length=1.0, seed=None):
        self.n_hash_funcs = n_hash_funcs
        self.window_length = window_length
        self.seed = seed

    def fit(self, X):
        """
        .

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.rng_ = np.random.default_rng(self.seed)
        self.X_ = np.array(
            [X[i].flatten() for i in range(len(X))]
        )  # n_samples, n_channels * n_timepoints

        self.window_length_ = max(1, int(self.X_.shape[1] * self.window_length))
        # Can replace with choice [-1, 1]
        self.hash_funcs_ = self.rng_.uniform(
            low=-1, high=1.0, size=(self.n_hash_funcs, self.window_length_)
        )
        self.start_points_ = self.rng_.choice(
            self.X_.shape[1] - self.window_length_ + 1,
            size=self.n_hash_funcs,
            replace=True,
        )

        bool_hashes = _collection_to_bool(
            self.X_, self.hash_funcs_, self.start_points_, self.window_length_
        )
        # could yield this
        str_hashes = [hash(bool_hashes[i].tobytes()) for i in range(len(bool_hashes))]
        self.dict_X_index = {}
        self.dict_bool_hashes = {}
        for i in range(len(str_hashes)):
            if str_hashes[i] in self.dict_X_index:
                self.dict_X_index[str_hashes[i]].append(i)
            else:
                self.dict_X_index[str_hashes[i]] = [i]
                self.dict_bool_hashes[str_hashes[i]] = bool_hashes[i]

        self.bool_hashes_value_list = np.asarray(list(self.dict_bool_hashes.values()))
        self.bool_hashes_key_list = np.asarray(list(self.dict_bool_hashes.keys()))

        return self

    def update(self, X):
        """
        .

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        X_ = np.array(
            [X[i].flatten() for i in range(len(X))]
        )  # n_samples, n_channels * n_timepoints
        bool_hashes = _collection_to_bool(
            X_, self.hash_funcs_, self.start_points_, self.window_length_
        )

        str_hashes = [hash(bool_hashes[i].tobytes()) for i in range(len(bool_hashes))]
        base_index = self.X_.shape[0]
        for i in range(len(str_hashes)):
            if str_hashes[i] in self.dict_X_index:
                self.dict_X_index[str_hashes[i]].append(i + base_index)
            else:
                self.dict_X_index[str_hashes[i]] = [i + base_index]
                self.dict_bool_hashes[str_hashes[i]] = bool_hashes[i]
        self.X_ = np.concatenate((self.X_, X_))

        self.bool_hashes_value_list = np.asarray(list(self.dict_bool_hashes.values()))
        self.bool_hashes_key_list = np.asarray(list(self.dict_bool_hashes.keys()))
        return self

    def get_bucket_collection_indexes(self, X):
        """
        .

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        bool_hash = _series_to_bool(
            X.flatten(), self.hash_funcs_, self.start_points_, self.window_length_
        )
        str_hash = hash(bool_hash.tobytes())
        if str_hash in self.dict_X_index:
            return self.dict_X_index[str_hash]
        else:
            return []

    def predict(self, X, k=1):
        """
        .

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        k : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        top_k : TYPE
            DESCRIPTION.

        """
        X_ = np.array([X[i].flatten() for i in range(len(X))])
        bool_hashes = _collection_to_bool(
            X_, self.hash_funcs_, self.start_points_, self.window_length_
        )
        top_k = np.zeros((len(X), k), dtype=int)
        dists = _hamming_dist_matrix(self.bool_hashes_value_list, bool_hashes)
        self.h_dists = dists
        # Deal with equality by merging bucket contents ?
        for i_x in range(len(X)):
            ids = np.argsort(dists[i_x])
            _i = 0
            c = k
            while c > 0:
                candidates = self.dict_X_index[self.bool_hashes_key_list[ids[_i]]]
                # Can do exact search by computing distances here
                if len(candidates) > c:
                    candidates = candidates[:c]
                top_k[i_x, k - c : k - c + len(candidates)] = candidates
                c -= len(candidates)
                _i += 1
        return top_k

    def find_motif(Index, X=None):
        """
        .

        Parameters
        ----------
        Index : TYPE
            DESCRIPTION.
        X : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        pass
