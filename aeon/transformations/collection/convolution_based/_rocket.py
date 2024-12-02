"""Rocket transformer."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["Rocket"]

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer, Normalizer
from aeon.utils.validation import check_n_jobs


class Rocket(BaseCollectionTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).

    A kernel (or convolution) is a subseries used to create features that can be used
    in machine learning tasks. ROCKET [1]_ generates a large number of random
    convolutional kernels in the fit method. The length and dilation of each kernel
    are also randomly generated. The kernels are used in the transform stage to
    generate a new set of features. A kernel is used to create an activation map for
    each series by running it across a time series, including random length and
    dilation. It transforms the time series with two features per kernel. The first
    feature is global max pooling and the second is proportion of positive values
    (or PPV).


    Parameters
    ----------
    n_kernels : int, default=10,000
       Number of random convolutional kernels.
    normalise : bool, default True
       Whether or not to normalise the input time series per instance.
    n_jobs : int, default=1
       The number of jobs to run in parallel for `transform`. ``-1`` means using all
       processors.
    random_state : None or int, optional, default = None
        Seed for random number generation.

    See Also
    --------
    MiniRocket, MultiRocket

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph
        and Webb, Geoffrey I,
        "ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels",2020,
      https://link.springer.com/article/10.1007/s10618-020-00701-z,
      https://arxiv.org/abs/1910.13051

    Examples
    --------
    >>> from aeon.transformations.collection.convolution_based import Rocket
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> trf = Rocket(n_kernels=512)
    >>> trf.fit(X_train)
    Rocket(n_kernels=512)
    >>> X_train = trf.transform(X_train)
    >>> X_test = trf.transform(X_test)
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        n_kernels=10_000,
        normalise=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        if isinstance(self.random_state, int):
            self._random_state = self.random_state
        else:
            self._random_state = None
        n_channels = X[0].shape[0]

        # The only use of n_timepoints is to set the maximum dilation
        self.fit_min_length_ = X[0].shape[1]
        self.kernels = _generate_kernels(
            self.fit_min_length_, self.n_kernels, n_channels, self._random_state
        )
        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray (n_cases, n_kernels), transformed features
        """
        if self.normalise:
            norm = Normalizer()
            X = norm.fit_transform(X)
        prev_threads = get_num_threads()

        n_jobs = check_n_jobs(self.n_jobs)

        set_num_threads(n_jobs)
        X_ = _apply_kernels(X, self.kernels)

        set_num_threads(prev_threads)
        return X_


@njit(fastmath=True, cache=True)
def _generate_kernels(n_timepoints, n_kernels, n_channels, seed):
    if seed is not None:
        np.random.seed(seed)
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, n_kernels).astype(np.int32)

    num_channel_indices = np.zeros(n_kernels, dtype=np.int32)
    for i in range(n_kernels):
        limit = min(n_channels, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(n_kernels, dtype=np.float32)
    dilations = np.zeros(n_kernels, dtype=np.int32)
    paddings = np.zeros(n_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(n_kernels):
        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _weights = np.random.normal(0, 1, _num_channel_indices * _length).astype(
            np.float32
        )

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0  # for weights (per channel)
        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_channels), _num_channel_indices, replace=False
        )

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((n_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )


@njit(
    parallel=True,
    fastmath=True,
    cache=True,
)
def _apply_kernels(X, kernels):
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        n_channel_indices,
        channel_indices,
    ) = kernels
    n_cases = len(X)
    n_channels, _ = X[0].shape
    n_kernels = len(lengths)

    _X = np.zeros((n_cases, n_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in prange(n_cases):
        a1 = 0  # for weights
        a2 = 0  # for channel_indices
        a3 = 0  # for features

        for j in range(n_kernels):
            b1 = a1 + n_channel_indices[j] * lengths[j]
            b2 = a2 + n_channel_indices[j]
            b3 = a3 + 2

            if n_channel_indices[j] == 1:
                _X[i][a3:b3] = _apply_kernel_univariate(
                    X[i][channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )

            else:
                _weights = weights[a1:b1].reshape((n_channel_indices[j], lengths[j]))

                _X[i][a3:b3] = _apply_kernel_multivariate(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    n_channel_indices[j],
                    channel_indices[a2:b2],
                )

            a1 = b1
            a2 = b2
            a3 = b3

    return _X.astype(np.float32)


@njit(fastmath=True, cache=True)
def _apply_kernel_univariate(X, weights, length, bias, dilation, padding):
    """Apply a single kernel to a univariate series."""
    n_timepoints = len(X)

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = -np.inf

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias

        index = i

        for j in range(length):
            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_length), np.float32(_max)


@njit(fastmath=True, cache=True)
def _apply_kernel_multivariate(
    X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices
):
    """Apply a kernel to a single multivariate time series."""
    n_columns, n_timepoints = X.shape

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = -np.inf
    end = (n_timepoints + padding) - ((length - 1) * dilation)
    for i in range(-padding, end):
        _sum = bias
        index = i
        for j in range(length):
            if index > -1 and index < n_timepoints:
                for k in range(num_channel_indices):
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]
            index = index + dilation
        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1
    return np.float32(_ppv / output_length), np.float32(_max)
