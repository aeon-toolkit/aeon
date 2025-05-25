"""SAST Transformer."""

from typing import Optional, Union

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.numba.general import z_normalise_series
from aeon.utils.validation import check_n_jobs


@njit(fastmath=False)
def _apply_kernel(ts: np.ndarray, arr: np.ndarray) -> float:
    d_best = np.inf  # sdist
    m = ts.shape[0]
    kernel = arr[~np.isnan(arr)]  # ignore nan

    kernel_len = kernel.shape[0]
    for i in range(m - kernel_len + 1):
        d = np.sum((z_normalise_series(ts[i : i + kernel_len]) - kernel) ** 2)
        if d < d_best:
            d_best = d

    return d_best


@njit(parallel=True, fastmath=True)
def _apply_kernels(X: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i] = _apply_kernel(ts, k)
    return out


class SAST(BaseCollectionTransformer):
    """Scalable and Accurate Subsequence Transform (SAST).

    SAST [1]_ first randomly selects k time series from each class (they are called
    reference time series). Then SAST generates all the subsequences of the
    specified lengths from these reference time series. These subsequences
    are then used to transform a time series dataset, replacing each time
    series by the vector of its distance to each subsequence.

    Parameters
    ----------
    lengths : int[], default = None
        an array containing the lengths of the subsequences
        to be generated. If None, will be inferred during fit
        as np.arange(3, X.shape[1])
    stride : int, default = 1
        the stride used when generating subsequences
    nb_inst_per_class : int, default = 1
        the number of reference time series to select per class
    seed : int, default = None
        the seed of the random generator
    n_jobs : int, default -1
        Number of threads to use for the transform.
        The available CPU count is used if this value is less than 1


    References
    ----------
    .. [1] Mbouopda, Michael Franklin, and Engelbert Mephu Nguifo.
    "Scalable and accurate subsequence transform for time series classification."
    Pattern Recognition 147 (2023): 110121.
    https://www.sciencedirect.com/science/article/abs/pii/S003132032300818X,
    https://uca.hal.science/hal-03087686/document

    Examples
    --------
    >>> from aeon.transformations.collection.shapelet_based import SAST
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> sast = SAST()
    >>> sast.fit(X_train, y_train)
    SAST()
    >>> X_train = sast.transform(X_train)
    >>> X_test = sast.transform(X_test)

    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": False,
        "capability:multithreading": True,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        lengths: Optional[np.ndarray] = None,
        stride: int = 1,
        nb_inst_per_class: int = 1,
        seed: Optional[int] = None,
        n_jobs: int = 1,  # Parallel processing
    ):
        super().__init__()
        self.lengths = lengths
        self.stride = stride
        self.nb_inst_per_class = nb_inst_per_class
        self._kernels = None  # z-normalized subsequences
        self._kernel_orig = None  # non z-normalized subsequences
        self._start_points = []  # To store the start positions
        self._classes = []  # To store the class of each shapelet
        self._source_series = []  # To store the index of the original time series
        self.kernels_generators_ = {}  # Reference time series
        self.n_jobs = n_jobs
        self.seed = seed

    def _fit(self, X: np.ndarray, y: Union[np.ndarray, list]) -> "SAST":
        """Select reference time series and generate subsequences from them.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : SAST
            This transformer

        """
        X_ = np.reshape(X, (X.shape[0], X.shape[-1]))
        self._length_list = (
            self.lengths if self.lengths is not None else np.arange(3, X_.shape[1])
        )

        self._random_state = (
            np.random.RandomState(self.seed)
            if not isinstance(self.seed, np.random.RandomState)
            else self.seed
        )

        classes = np.unique(y)
        self._num_classes = classes.shape[0]
        class_values_of_candidates = []
        candidates_ts = []
        source_series_indices = []  # List to store original indices

        for c in classes:
            X_c = X_[y == c]

            # convert to int because if self.
            # nb_inst_per_class is float, the result of np.min() will be float
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            choosen = self._random_state.permutation(X_c.shape[0])[:cnt]
            candidates_ts.append(X_c[choosen])
            self.kernels_generators_[c] = X_c[choosen]
            class_values_of_candidates.extend([c] * cnt)
            source_series_indices.extend(
                np.where(y == c)[0][choosen]
            )  # Record the original indices

        candidates_ts = np.concatenate(candidates_ts, axis=0)

        self._length_list = self._length_list[self._length_list <= X_.shape[1]]

        max_shp_length = max(self._length_list)

        n, m = candidates_ts.shape

        n_kernels = n * np.sum([m - len_ + 1 for len_ in self._length_list])

        self._kernels = np.full(
            (n_kernels, max_shp_length), dtype=np.float32, fill_value=np.nan
        )
        self._kernel_orig = []
        self._start_points = []  # Reset start positions
        self._classes = []  # Reset class information
        self._source_series = []  # Reset source series information

        k = 0
        for shp_length in self._length_list:
            for i in range(candidates_ts.shape[0]):
                for j in range(0, candidates_ts.shape[1] - shp_length + 1, self.stride):
                    end = j + shp_length
                    can = np.squeeze(candidates_ts[i][j:end])
                    self._kernel_orig.append(can)
                    self._kernels[k, :shp_length] = z_normalise_series(can)
                    self._start_points.append(j)  # Store the start position
                    self._classes.append(
                        class_values_of_candidates[i]
                    )  # Store the class of the shapelet
                    self._source_series.append(
                        source_series_indices[i]
                    )  # Store the original index of the time series
                    k += 1
        return self

    def _transform(
        self, X: np.ndarray, y: Optional[Union[np.ndarray, list]] = None
    ) -> np.ndarray:
        """Transform the input X using the generated subsequences.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            Ignored argument, interface compatibility

        Returns
        -------
        X_transformed: np.ndarray shape (n_cases, n_timepoints),
            The transformed data
        """
        X_ = np.reshape(X, (X.shape[0], X.shape[-1]))

        prev_threads = get_num_threads()

        n_jobs = check_n_jobs(self.n_jobs)

        set_num_threads(n_jobs)
        X_transformed = _apply_kernels(X_, self._kernels)  # subsequence transform of X
        set_num_threads(prev_threads)

        return X_transformed
