"""Multivariate MiniRocket transformer."""

__maintainer__ = []
__all__ = ["MiniRocketMultivariateVariable"]

import multiprocessing
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange, set_num_threads, vectorize

from aeon.transformations.collection import BaseCollectionTransformer


class MiniRocketMultivariateVariable(BaseCollectionTransformer):
    """MINIROCKET (Multivariate, unequal length).

    MINImally RandOm Convolutional KErnel Transform. [1]_

    **Multivariate** and **unequal length**

    A provisional and naive extension of MINIROCKET to multivariate input
    with unequal length provided by the authors [2]_. For better
    performance, use the aeon class MiniRocket for univariate input,
    and MiniRocketMultivariate for equal-length multivariate input.

    Parameters
    ----------
    num_kernels : int, default=10,000
       Number of random convolutional kernels. The calculated number of features is the
       nearest multiple of ``n_features_per_kernel(default 4)*84=336 < 50,000``
       (``2*n_features_per_kernel(default 4)*num_kernels(default 10,000)``).
    max_dilations_per_kernel : int, default=32
        Maximum number of dilations per kernel.
    reference_length : int or str, default = `'max'`
        Series-length of reference, str defines how to infer from `X` during `fit`.
        Options are `'max'`, `'mean'`, `'median'`, `'min'`.
    pad_value_short_series : float or None, default=None
        Whether padding series with length less than 9 to value. If None, no
        padding is performed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. ``-1`` means using all
        processors.
    random_state : None or int, default = None
        Seed for random number generation.

    Examples
    --------
    >>> from aeon.transformations.collection.convolution_based import (
    ...     MiniRocketMultivariateVariable
    ... )
    >>> from aeon.datasets import load_japanese_vowels
    >>> # load multivariate and unequal length dataset
    >>> X_train, _ = load_japanese_vowels(split="train")
    >>> X_test, _ = load_japanese_vowels(split="test")
    >>> pre_clf = MiniRocketMultivariateVariable(pad_value_short_series=0.0)
    >>> pre_clf.fit(X_train, y=None)
    MiniRocketMultivariateVariable(pad_value_short_series=0.0)
    >>> X_transformed = pre_clf.transform(X_test)
    >>> X_transformed.shape
    (370, 9996)

    Raises
    ------
    ValueError
        If any multivariate n_timepoints in X is < 9 and
        pad_value_short_series is set to None

    See Also
    --------
    MultiRocket, MiniRocket, MiniRocketMultivariate, Rocket

    References
    ----------
    .. [1] Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
        MINIROCKET: A Very Fast (Almost) Deterministic Transform for
        Time Series Classification, 2020, arXiv:2012.08791

    .. [2] Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
           https://github.com/angus924/minirocket

    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "X_inner_type": "np-list",
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        num_kernels=10000,
        max_dilations_per_kernel=32,
        reference_length="max",
        pad_value_short_series=None,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.reference_length = reference_length
        self._fitted_reference_length = None
        self.pad_value_short_series = pad_value_short_series

        self.n_jobs = n_jobs
        self.random_state = random_state

        if random_state is None:
            self.random_state_ = random_state
        elif isinstance(random_state, int):
            self.random_state_ = np.int32(random_state)
        else:
            raise ValueError(
                f"random_state in MiniRocketMultivariateVariable must be int or None, "
                f"but found <{type(random_state)} {random_state}>"
            )

        self._reference_modes = ["max", "mean", "median", "min"]
        if not (isinstance(reference_length, int) and reference_length >= 9) and not (
            isinstance(reference_length, str)
            and (reference_length in self._reference_modes)
        ):
            raise ValueError(
                "reference_length in MiniRocketMultivariateVariable must be int>=9 or "
                "'max', 'mean', 'median', but found reference_length="
                f"{reference_length}"
            )

        super().__init__()

    def _fit(self, X, y=None):
        """Fits dilations and biases to input time series.

        Parameters
        ----------
        X : List of 2D np.ndarray
        y : ignored argument for interface compatibility

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If any multivariate n_timepoints in X is < 9 and
            pad_value_short_series is set to None
        """
        X_2d_t, lengths_1darray = _np_list_transposed2D_array_and_len_list(
            X, pad=self.pad_value_short_series
        )

        if isinstance(self.reference_length, int):
            _reference_length = self.reference_length
        elif self.reference_length in self._reference_modes:
            # np.mean, np.max, np.median, np.min ..
            _reference_length = getattr(np, self.reference_length)(lengths_1darray)
        else:
            raise ValueError(
                "reference_length in MiniRocketMultivariateVariable must be int>=9 or "
                "'max', 'mean', 'median', but found reference_length="
                f"{self.reference_length}"
            )
        self._fitted_reference_length = int(max(9, _reference_length))

        if lengths_1darray.min() < 9:
            failed_index = np.where(lengths_1darray < 9)[0]
            raise ValueError(
                f"X must be >= 9 for all samples, but found miniumum to be "
                f"{lengths_1darray.min()}; at index {failed_index}, pad shorter "
                "series so that n_timepoints >= 9 for all samples."
            )

        if lengths_1darray.min() == lengths_1darray.max():
            warnings.warn(
                "X is of equal length, consider using MiniRocketMultivariate for "
                "speedup and stability instead.",
                stacklevel=2,
            )
        if X_2d_t.shape[0] == 1:
            warnings.warn(
                "X is univariate, consider using the univariate MiniRocket for "
                "speedup and stability instead.",
                stacklevel=2,
            )

        self.parameters = _fit_multi_var(
            X_2d_t,
            L=lengths_1darray,
            reference_length=self._fitted_reference_length,
            num_features=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            seed=self.random_state_,
        )
        return self

    def _transform(self, X, y=None):
        """Transform input time series.

        Parameters
        ----------
        X : 2D list on np.ndarray
        y : ignored argument for interface compatibility

        Returns
        -------
           np.ndarray, size (n_cases, num_kernels)

        Raises
        ------
        ValueError
            If any multivariate n_timepoints in X is < 9 and
            pad_value_short_series is set to None
        """
        X_2d_t, L = _np_list_transposed2D_array_and_len_list(
            X, pad=self.pad_value_short_series
        )
        # change n_jobs dependend on value and existing cores
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        X_ = _transform_multi_var(X_2d_t, L, self.parameters)
        set_num_threads(prev_threads)
        return X_


def _np_list_transposed2D_array_and_len_list(
    X: List[pd.DataFrame], pad: Union[int, float, None] = 0
):
    """Convert a list of 2D numpy to a 2D array and a list of lengths.

    Parameters
    ----------
    X : List of dataframes
        List of length n_cases, with
        dataframes of n_timepoints-rows and n_channels-columns
    pad : float or None. if float/int,pads multivariate series with 'pad',
        so that each series has at least length 9.
        if None, no padding is applied.

    Returns
    -------
    np.ndarray: 2D array of shape =
        [n_channels, sum(length_series(i) for i in n_cases)],
        np.float32
    np.ndarray: 1D array of shape = [n_cases]
        with length of each series, np.int32

    Raises
    ------
    ValueError
        If any multivariate n_timepoints in X is < 9 and
        pad_value_short_series is set to None
    """
    vec = []
    lengths = []

    for _x in X:
        _x_shape = _x.shape
        if _x_shape[1] < 9:
            if pad is not None:
                # emergency: pad with zeros up to 9.
                lengths.append(9)
                padding_width = ((0, 0), (0, 9 - _x_shape[1]))
                x_pad = np.pad(_x, padding_width, mode="constant", constant_values=pad)
                vec.append(np.transpose(x_pad))
            else:
                raise ValueError(
                    "X n_timepoints must be >= 9 for all samples"
                    f"but sample with n_timepoints {_x_shape[1]} found. Consider"
                    " padding, discard, or setting a pad_value_short_series value"
                )
        else:
            lengths.append(_x_shape[1])
            vec.append(np.transpose(_x))

    X_2d_t = np.vstack(vec).T.astype(dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)

    if not lengths.sum() == X_2d_t.shape[1]:
        raise ValueError("X_new and lengths do not match. check input dimension")

    return X_2d_t, lengths


# code below from the orignal authors: https://github.com/angus924/minirocket


@njit(
    "float32[:](float32[:,:],int32[:],int32[:],int32[:],int32[:],int32[:],float32[:],"
    "optional(int32))",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_multi_var(
    X,
    L,
    num_channels_per_combination,
    channel_indices,
    dilations,
    num_features_per_dilation,
    quantiles,
    seed,
):
    if seed is not None:
        np.random.seed(seed)
    n_cases = len(L)

    num_channels, _ = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array(
    # >>>    [_ for _ in combinations(np.arange(9), 3)], dtype = np.int32
    # >>> )
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[
                num_channels_start:num_channels_end
            ]

            example_index = np.random.randint(n_cases)

            input_length = np.int64(L[example_index])

            b = np.sum(L[0 : example_index + 1])
            a = b - input_length

            _X = X[channels_this_combination, a:b]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(
                (num_channels_this_combination, input_length), dtype=np.float32
            )
            C_alpha[:] = A

            C_gamma = np.zeros(
                (9, num_channels_this_combination, input_length), dtype=np.float32
            )
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                # thanks to Murtaza Jafferji @murtazajafferji for suggesting this fix
                if end > 0:
                    C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                    C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                if start < input_length:
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
            num_channels_start = num_channels_end

    return biases


def _fit_dilations_multi_var(reference_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((reference_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles_multi_var(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def _fit_multi_var(
    X,
    L,
    reference_length: int,
    num_features=10_000,
    max_dilations_per_kernel=32,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    # note in relation to dilation:
    # * change *reference_length* according to what is appropriate for your
    #   application, e.g., L.max(), L.mean(), np.median(L)
    # * use _fit_multi_var(...) with an appropriate subset of time series, e.g., for
    #   reference_length = L.mean(), call _fit_multi_var(...) using only time series
    #   of at least length L.mean() [see filter_by_length(...)]
    if reference_length is None:
        raise ValueError("reference_length must be specified")

    num_channels, _ = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations_multi_var(
        reference_length, num_features, max_dilations_per_kernel
    )

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles_multi_var(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (
        2 ** np.random.uniform(0, max_exponent, num_combinations)
    ).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(
            num_channels, num_channels_this_combination, replace=False
        )

        num_channels_start = num_channels_end

    biases = _fit_biases_multi_var(
        X,
        L,
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        quantiles,
        seed,
    )

    return (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    )


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0


@njit(
    "float32[:,:](float32[:,:],int32[:],Tuple((int32[:],int32[:],int32[:],int32[:],"
    "float32[:])))",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi_var(X, L, parameters):
    n_cases = len(L)

    num_channels, _ = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array(
    # >>>     [_ for _ in combinations(np.arange(9), 3)], dtype = np.int32
    # >>> )
    indices = np.array(
        (
            0,
            1,
            2,
            0,
            1,
            3,
            0,
            1,
            4,
            0,
            1,
            5,
            0,
            1,
            6,
            0,
            1,
            7,
            0,
            1,
            8,
            0,
            2,
            3,
            0,
            2,
            4,
            0,
            2,
            5,
            0,
            2,
            6,
            0,
            2,
            7,
            0,
            2,
            8,
            0,
            3,
            4,
            0,
            3,
            5,
            0,
            3,
            6,
            0,
            3,
            7,
            0,
            3,
            8,
            0,
            4,
            5,
            0,
            4,
            6,
            0,
            4,
            7,
            0,
            4,
            8,
            0,
            5,
            6,
            0,
            5,
            7,
            0,
            5,
            8,
            0,
            6,
            7,
            0,
            6,
            8,
            0,
            7,
            8,
            1,
            2,
            3,
            1,
            2,
            4,
            1,
            2,
            5,
            1,
            2,
            6,
            1,
            2,
            7,
            1,
            2,
            8,
            1,
            3,
            4,
            1,
            3,
            5,
            1,
            3,
            6,
            1,
            3,
            7,
            1,
            3,
            8,
            1,
            4,
            5,
            1,
            4,
            6,
            1,
            4,
            7,
            1,
            4,
            8,
            1,
            5,
            6,
            1,
            5,
            7,
            1,
            5,
            8,
            1,
            6,
            7,
            1,
            6,
            8,
            1,
            7,
            8,
            2,
            3,
            4,
            2,
            3,
            5,
            2,
            3,
            6,
            2,
            3,
            7,
            2,
            3,
            8,
            2,
            4,
            5,
            2,
            4,
            6,
            2,
            4,
            7,
            2,
            4,
            8,
            2,
            5,
            6,
            2,
            5,
            7,
            2,
            5,
            8,
            2,
            6,
            7,
            2,
            6,
            8,
            2,
            7,
            8,
            3,
            4,
            5,
            3,
            4,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            3,
            5,
            6,
            3,
            5,
            7,
            3,
            5,
            8,
            3,
            6,
            7,
            3,
            6,
            8,
            3,
            7,
            8,
            4,
            5,
            6,
            4,
            5,
            7,
            4,
            5,
            8,
            4,
            6,
            7,
            4,
            6,
            8,
            4,
            7,
            8,
            5,
            6,
            7,
            5,
            6,
            8,
            5,
            7,
            8,
            6,
            7,
            8,
        ),
        dtype=np.int32,
    ).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_cases, num_features), dtype=np.float32)

    for example_index in prange(n_cases):
        input_length = np.int64(L[example_index])

        b = np.sum(L[0 : example_index + 1])
        a = b - input_length

        _X = X[:, a:b]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                # thanks to Murtaza Jafferji @murtazajafferji for suggesting this fix
                if end > 0:
                    C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                    C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                if start < input_length:
                    C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                    C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[
                    combination_index
                ]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[
                    num_channels_start:num_channels_end
                ]

                index_0, index_1, index_2 = indices[kernel_index]

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                for feature_count in range(num_features_this_dilation):
                    features[example_index, feature_index_start + feature_count] = _PPV(
                        C, biases[feature_index_start + feature_count]
                    ).mean()

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features
