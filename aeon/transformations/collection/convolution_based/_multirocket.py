import multiprocessing
from itertools import combinations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer


class MultiRocket(BaseCollectionTransformer):
    """Multi RandOm Convolutional KErnel Transform (MultiRocket).

    MultiRocket [1]_ uses the same set of kernels as MiniRocket on both the raw
    series and the first order differenced series representation. It uses a different
    set of dilations and used for each representation. In addition to percentage of
    positive values (PPV), MultiRocket adds 3 pooling operators: Mean of Positive
    Values (MPV); Mean of Indices of Positive Values (MIPV); and Longest Stretch of
    Positive Values (LSPV).

    Parameters
    ----------
    n_kernels : int, default = 6,250
       Number of random convolutional kernels. The calculated number of features is the
       nearest multiple of ``n_features_per_kernel(default 4)*84=336 < 50,000``
       (``2*n_features_per_kernel(default 4)*n_kernels(default 6,250)``).
    max_dilations_per_kernel : int, default = 32
        Maximum number of dilations per kernel.
    n_features_per_kernel : int, default = 4
        Number of features per kernel.
    normalise : bool, default False
        Whether or not to normalise the input time series per instance.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. ``-1`` means using all
        processors.
    random_state : None or int, default = None
        Seed for random number generation.

    Attributes
    ----------
    parameter : tuple
        Parameter (dilations, n_features_per_dilation, biases) for
        transformation of input `X`.
    parameter1 : tuple
        Parameter (dilations, n_features_per_dilation, biases) for
        transformation of input ``X1 = np.diff(X, 1)``.


    See Also
    --------
    MiniRocket, Rocket

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph and
    Webb, Geoffrey I, "MultiRocket: Multiple pooling operators and transformations
    for fast and effective time series classification",2022,
    https://link.springer.com/article/10.1007/s10618-022-00844-1
    https://arxiv.org/abs/2102.00457

    Examples
    --------
     >>> from aeon.transformations.collection.convolution_based import MultiRocket
     >>> from aeon.datasets import load_unit_test
     >>> X_train, y_train = load_unit_test(split="train")
     >>> X_test, y_test = load_unit_test(split="test")
     >>> trf = MultiRocket(n_kernels=512)
     >>> trf.fit(X_train)
     MultiRocket(n_kernels=512)
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
        n_kernels=6_250,
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        normalise=False,
        n_jobs=1,
        random_state=None,
    ):
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

        self.n_kernels = n_kernels

        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.parameter = None
        self.parameter1 = None

        super().__init__()

    def _fit(self, X, y=None):
        """Fit dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        self.random_state_ = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )
        if self.random_state_ is not None:
            np.random.seed(self.random_state_)

        _, n_channels, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                f"n_timepoints must be >= 9, but found {n_timepoints};"
                " zero pad shorter series so that n_timepoints == 9"
            )
        X = X.astype(np.float32)
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        if n_channels == 1:
            X = X.squeeze()
            self.parameter = self._fit_univariate(X)
            _X1 = np.diff(X, 1)
            self.parameter1 = self._fit_univariate(_X1)
        else:
            self.parameter = self._fit_multivariate(X)
            _X1 = np.diff(X, 1)
            self.parameter1 = self._fit_multivariate(_X1)

        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform.
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        _, n_channels, n_timepoints = X.shape
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)

        X = X.astype(np.float32)
        if n_channels > 1:
            X1 = np.diff(X, 1)
            X = _transform_multi(
                X,
                X1,
                self.parameter,
                self.parameter1,
                self.n_features_per_kernel,
                MultiRocket._indices,
                self.random_state_,
            )
        else:
            X = X.reshape(X.shape[0], X.shape[2])
            X1 = np.diff(X, 1)
            X = _transform_uni(
                X,
                X1,
                self.parameter,
                self.parameter1,
                self.n_features_per_kernel,
                MultiRocket._indices,
                self.random_state_,
            )

        X = np.nan_to_num(X)  # not sure about this!

        set_num_threads(prev_threads)
        return X

    def _fit_univariate(self, X):
        _, input_length = X.shape

        n_kernels = 84

        dilations, n_features_per_dilation = _fit_dilations(
            input_length, self.n_kernels, self.max_dilations_per_kernel
        )

        n_features_per_kernel = np.sum(n_features_per_dilation)

        quantiles = _quantiles(n_kernels * n_features_per_kernel)

        biases = _fit_biases_univariate(
            X,
            dilations,
            n_features_per_dilation,
            quantiles,
            MultiRocket._indices,
            self.random_state_,
        )

        return dilations, n_features_per_dilation, biases

    def _fit_multivariate(self, X):
        _, n_channels, input_length = X.shape

        n_kernels = 84

        dilations, n_features_per_dilation = _fit_dilations(
            input_length, self.n_kernels, self.max_dilations_per_kernel
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

        biases = _fit_biases_multivariate(
            X,
            n_channels_per_combination,
            channel_indices,
            dilations,
            n_features_per_dilation,
            quantiles,
            MultiRocket._indices,
            self.random_state_,
        )

        return (
            n_channels_per_combination,
            channel_indices,
            dilations,
            n_features_per_dilation,
            biases,
        )


@njit(
    "float32[:,:](float32[:,:],float32[:,:],Tuple((int32[:],int32[:],float32[:])),"
    "Tuple((int32[:],int32[:],float32[:])),int32, int32[:,:],optional(int32))",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_uni(
    X, X1, parameters, parameters1, n_features_per_kernel, indices, seed
):
    if seed is not None:
        np.random.seed(seed)
    n_cases, n_timepoints = X.shape

    dilations, n_features_per_dilation, biases = parameters
    dilations1, n_features_per_dilation1, biases1 = parameters1
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_dilations1 = len(dilations1)

    n_features = n_kernels * np.sum(n_features_per_dilation)
    n_features1 = n_kernels * np.sum(n_features_per_dilation1)

    features = np.zeros(
        (n_cases, (n_features + n_features1) * n_features_per_kernel),
        dtype=np.float32,
    )
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(n_cases):
        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        for dilation_index in range(n_dilations):
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            n_features_this_dilation = n_features_per_dilation[dilation_index]

            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )

                feature_index_start = feature_index_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(n_dilations1):
            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            n_features_this_dilation = n_features_per_dilation1[dilation_index]

            C_alpha = np.zeros(n_timepoints - 1, dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, n_timepoints - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A1[:end]
                C_gamma[gamma_index, -end:] = G1[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A1[start:]
                C_gamma[gamma_index, :-start] = G1[start:]

                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )

                feature_index_start = feature_index_end

    return features


@njit(
    "float32[:,:](float32[:,:,:],float32[:,:,:],"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),int32, int32[:,:],"
    "optional(int32))",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi(
    X, X1, parameters, parameters1, n_features_per_kernel, indices, seed
):
    n_cases, n_channels, n_timepoints = X.shape
    (
        n_channels_per_combination,
        channel_indices,
        dilations,
        n_features_per_dilation,
        biases,
    ) = parameters
    if seed is not None:
        np.random.seed(seed)

    _, _, dilations1, n_features_per_dilation1, biases1 = parameters1
    n_kernels = len(indices)
    n_dilations = len(dilations)
    n_dilations1 = len(dilations1)

    n_features = n_kernels * np.sum(n_features_per_dilation)
    n_features1 = n_kernels * np.sum(n_features_per_dilation1)

    features = np.zeros(
        (n_cases, (n_features + n_features1) * n_features_per_kernel),
        dtype=np.float32,
    )
    n_features_per_transform = np.int64(features.shape[1] / 2)
    for example_index in prange(n_cases):
        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        combination_index = 0
        n_channels_start = 0

        for dilation_index in range(n_dilations):
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            n_features_this_dilation = n_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_channels, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_channels, n_timepoints), dtype=np.float32)
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

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                n_channels_this_combination = n_channels_per_combination[
                    combination_index
                ]

                n_channels_end = n_channels_start + n_channels_this_combination

                channels_this_combination = channel_indices[
                    n_channels_start:n_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )

                feature_index_start = feature_index_end

                combination_index += 1
                n_channels_start = n_channels_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        n_channels_start = 0

        for dilation_index in range(n_dilations1):
            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            n_features_this_dilation = n_features_per_dilation1[dilation_index]

            C_alpha = np.zeros((n_channels, n_timepoints - 1), dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, n_channels, n_timepoints - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A1[:, :end]
                C_gamma[gamma_index, :, -end:] = G1[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A1[:, start:]
                C_gamma[gamma_index, :, :-start] = G1[:, start:]

                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                n_channels_this_combination = n_channels_per_combination[
                    combination_index
                ]

                n_channels_end = n_channels_start + n_channels_this_combination

                channels_this_combination = channel_indices[
                    n_channels_start:n_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + n_features
                        features[example_index, end] = max_stretch
                        end = end + n_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + n_features
                        features[example_index, end] = (
                            mean_index / ppv if ppv > 0 else -1
                        )

                feature_index_start = feature_index_end

    return features


@njit(
    "float32[:](float32[:,:],int32[:],int32[:],float32[:], int32[:,:],optional(int32))",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_univariate(
    X, dilations, n_features_per_dilation, quantiles, indices, seed
):
    if seed is not None:
        np.random.seed(seed)

    n_cases, input_length = X.shape
    n_kernels = len(indices)
    n_dilations = len(dilations)

    n_features = n_kernels * np.sum(n_features_per_dilation)

    biases = np.zeros(n_features, dtype=np.float32)

    feature_index_start = 0

    for dilation_index in range(n_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        n_features_this_dilation = n_features_per_dilation[dilation_index]

        for kernel_index in range(n_kernels):
            feature_index_end = feature_index_start + n_features_this_dilation

            _X = X[np.random.randint(n_cases)]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end

    return biases


@njit(
    "float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:], "
    "int32[:,:],optional(int32))",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_multivariate(
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

    n_cases, n_channels, input_length = X.shape

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
                (n_channels_this_combination, input_length), dtype=np.float32
            )
            C_alpha[:] = A

            C_gamma = np.zeros(
                (9, n_channels_this_combination, input_length), dtype=np.float32
            )
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

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


def _fit_dilations(input_length, n_features, max_dilations_per_kernel):
    n_kernels = 84

    n_features_per_kernel = n_features // n_kernels
    true_max_dilations_per_kernel = min(n_features_per_kernel, max_dilations_per_kernel)
    multiplier = n_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
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


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )
