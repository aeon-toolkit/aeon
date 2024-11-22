"""MiniRocket transformer."""

__maintainer__ = []
__all__ = ["MiniRocket"]

import multiprocessing
from itertools import combinations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads, vectorize

from aeon.transformations.collection import BaseCollectionTransformer


class MiniRocket(BaseCollectionTransformer):
    """MINImally RandOm Convolutional KErnel Transform (MiniRocket).

     MiniRocket [1]_ is an almost deterministic version of Rocket. It creates
     convolutions of length 9 with weights restricted to two values, and uses 84 fixed
     convolutions with six of one weight, three of the second weight to seed dilations.


    Parameters
    ----------
     n_kernels : int, default=10,000
        Number of random convolutional kernels. The number of kernels used is rounded
        down to the nearest multiple of 84, unless a value of less than 84 is passec,
        in which case it is set to 84.
     max_dilations_per_kernel : int, default=32
         Maximum number of dilations per kernel.
     n_jobs : int, default=1
         The number of jobs to run in parallel for `transform`. ``-1`` means using all
         processors.
     random_state : None or int, default = None
         Seed for random number generation.

    Attributes
    ----------
     self.parameters : Tuple (int32[:], int32[:], int32[:], int32[:], float32[:])
         n_channels_per_comb, channel_indices, dilations, n_features_per_dilation,
         biases

    See Also
    --------
     Rocket, MultiRocket, Hydra

    References
    ----------
     .. [1] Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I,
         "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series
         Classification",2020,
         https://dl.acm.org/doi/abs/10.1145/3447548.3467231,
         https://arxiv.org/abs/2012.08791

    Notes
    -----
     Directly adapted from the original implementation
     https://github.com/angus924/minirocket.

    Examples
    --------
     >>> from aeon.transformations.collection.convolution_based import MiniRocket
     >>> from aeon.datasets import load_unit_test
     >>> X_train, y_train = load_unit_test(split="train")
     >>> X_test, y_test = load_unit_test(split="test")
     >>> trf = MiniRocket(n_kernels=512)
     >>> trf.fit(X_train)
     MiniRocket(n_kernels=512)
     >>> X_train = trf.transform(X_train)
     >>> X_test = trf.transform(X_test)
    """

    _tags = {
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "capability:multivariate": True,
        "capability:multithreading": True,
    }
    # indices for the 84 kernels used by MiniRocket
    _indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype=np.int32)

    def __init__(
        self,
        n_kernels=10_000,
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fits dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        random_state = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )
        _, n_channels, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                f"n_timepoints must be >= 9, but found {n_timepoints};"
                " zero pad shorter series so that n_timepoints == 9"
            )
        X = X.astype(np.float32)
        if self.n_kernels < 84:
            self.n_kernels_ = 84
        else:
            self.n_kernels_ = self.n_kernels
        self.parameters = _static_fit(
            X, self.n_kernels_, self.max_dilations_per_kernel, random_state
        )
        return self

    def _transform(self, X, y=None):
        """Transform input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        X = X.astype(np.float32)
        _, n_channels, n_timepoints = X.shape
        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        if n_channels == 1:
            X = X.squeeze(1)
            X_ = _static_transform_uni(X, self.parameters, MiniRocket._indices)
        else:
            X_ = _static_transform_multi(X, self.parameters, MiniRocket._indices)
        set_num_threads(prev_threads)
        return X_


def _fit_dilations(n_timepoints, n_features, max_dilations_per_kernel):
    n_kernels = 84
    n_features_per_kernel = n_features // n_kernels
    true_max_dilations_per_kernel = min(n_features_per_kernel, max_dilations_per_kernel)
    multiplier = n_features_per_kernel / true_max_dilations_per_kernel
    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, n_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    n_features_per_dilation = (n_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector
    remainder = n_features_per_kernel - np.sum(n_features_per_dilation)
    i = 0
    while remainder > 0:
        n_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(n_features_per_dilation)
    return dilations, n_features_per_dilation


def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _static_fit(X, n_features=10_000, max_dilations_per_kernel=32, seed=None):
    if seed is not None:
        np.random.seed(seed)
    _, n_channels, n_timepoints = X.shape
    n_kernels = 84
    dilations, n_features_per_dilation = _fit_dilations(
        n_timepoints, n_features, max_dilations_per_kernel
    )
    n_features_per_kernel = np.sum(n_features_per_dilation)
    quantiles = _quantiles(n_kernels * n_features_per_kernel)
    n_dilations = len(dilations)
    n_combinations = n_kernels * n_dilations
    max_n_channels = min(n_channels, 9)
    max_exponent = np.log2(max_n_channels + 1)
    n_channels_per_combination = (
        2 ** np.random.uniform(0, max_exponent, n_combinations)
    ).astype(np.int32)
    channel_indices = np.zeros(n_channels_per_combination.sum(), dtype=np.int32)
    n_channels_start = 0
    for combination_index in range(n_combinations):
        n_channels_this_combination = n_channels_per_combination[combination_index]
        n_channels_end = n_channels_start + n_channels_this_combination
        channel_indices[n_channels_start:n_channels_end] = np.random.choice(
            n_channels, n_channels_this_combination, replace=False
        )
        n_channels_start = n_channels_end
    biases = _fit_biases(
        X,
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        quantiles,
        MiniRocket._indices,
        seed,
    )
    return (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    )


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    return 0


@njit(
    "float32[:,:](float32[:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32["
    ":])), int32[:,:])",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _static_transform_uni(X, parameters, indices):
    """Transform a 2D collection of univariate time series.

    Implemented separately to the multivariate version for numba efficiency reasons.
    See issue #1778.
    """
    n_cases, n_timepoints = X.shape
    (
        _,
        _,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    n_kernels = len(indices)
    n_dilations = len(dilations)
    f = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, f), dtype=np.float32)
    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        f_start = 0
        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = (8 * dilation) // 2
            n_features = n_features_per_dilation[j]
            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[4] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]
                end += dilation
            for gamma_index in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]
                start += dilation
            for k in range(n_kernels):
                f_end = f_start + n_features
                _padding1 = (_padding0 + k) % 2
                a, b, c = indices[k]
                C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]
                if _padding1 == 0:
                    for f in range(n_features):
                        features[i, f_start + f] = _PPV(C, biases[f_start + f]).mean()
                else:
                    for f in range(n_features):
                        features[i, f_start + f] = _PPV(
                            C[padding:-padding], biases[f_start + f]
                        ).mean()

                f_start = f_end
    return features


@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32["
    ":])), int32[:,:])",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _static_transform_multi(X, parameters, indices):
    n_cases, n_channels, n_timepoints = X.shape
    (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    features = np.zeros((n_cases, n_features), dtype=np.float32)
    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        f_start = 0
        comb = 0
        n_channels_start = 0
        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = (8 * dilation) // 2
            n_features_this_dilation = n_features_per_dilation[j]
            C_alpha = np.zeros((n_channels, n_timepoints), dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_channels, n_timepoints), dtype=np.float32)
            C_gamma[4] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation

            for gamma_index in range(5, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            for kernel_index in range(n_kernels):
                f_end = f_start + n_features_this_dilation
                n_channels_this_combo = n_channels_per_combination[comb]
                n_channels_end = n_channels_start + n_channels_this_combo
                channels_this_combo = channel_indices[n_channels_start:n_channels_end]
                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = indices[kernel_index]
                C = (
                    C_alpha[channels_this_combo]
                    + C_gamma[index_0][channels_this_combo]
                    + C_gamma[index_1][channels_this_combo]
                    + C_gamma[index_2][channels_this_combo]
                )
                C = np.sum(C, axis=0)
                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        features[i, f_start + feature_count] = _PPV(
                            C, biases[f_start + feature_count]
                        ).mean()
                else:
                    for feature_count in range(n_features_this_dilation):
                        features[i, f_start + feature_count] = _PPV(
                            C[padding:-padding],
                            biases[f_start + feature_count],
                        ).mean()
                f_start = f_end
                comb += 1
                n_channels_start = n_channels_end
    return features


@njit(
    "float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],"
    "int32[:,:],optional(int32))",  # noqa
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(
    X,
    n_channels_per_combination,
    channel_indices,
    dilations,
    n_features_per_dilation,
    quantiles,
    indices,
    seed,
):
    if seed is not None:
        np.random.seed(seed)
    n_cases, n_columns, n_timepoints = X.shape
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_features = n_kernels * np.sum(n_features_per_dilation)
    biases = np.zeros(n_features, dtype=np.float32)
    feature_index_start = 0
    combination_index = 0
    n_channels_start = 0
    for dilation_index in range(n_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2
        n_features_this_dilation = n_features_per_dilation[dilation_index]
        for kernel_index in range(n_kernels):
            feature_index_end = feature_index_start + n_features_this_dilation
            n_channels_this_combination = n_channels_per_combination[combination_index]
            n_channels_end = n_channels_start + n_channels_this_combination
            channels_this_combination = channel_indices[n_channels_start:n_channels_end]
            _X = X[np.random.randint(n_cases)][channels_this_combination]
            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X
            C_alpha = np.zeros(
                (n_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_alpha[:] = A
            C_gamma = np.zeros(
                (9, n_channels_this_combination, n_timepoints), dtype=np.float32
            )
            C_gamma[9 // 2] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation
            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation
            index_0, index_1, index_2 = indices[kernel_index]
            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)
            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )
            feature_index_start = feature_index_end
            combination_index += 1
            n_channels_start = n_channels_end
    return biases
