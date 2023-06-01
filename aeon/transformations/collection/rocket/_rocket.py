# -*- coding: utf-8 -*-
"""Rocket transformer."""

__author__ = "angus924"
__all__ = ["Rocket"]

import multiprocessing

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.base import BaseTransformer


class Rocket(BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).

    A kernel (or convolution) is a subseries used to create features that can be used
    in machine learning tasks. ROCKET [1]_  generates a large number of random
    convolutional kernels in the fit method. The length and dilation of each kernel
    are also randomly generated. The kernels are use in the transform stage to
    generate a new set of features. A kernel is used to create an activation map for
    each series by running it across a time series, including random length and
    dilation. It transforms the time series with two features per kernel. The first
    feature is global max pooling and the second is proportion of positive values.


    Parameters
    ----------
    num_kernels : int, default=10,000
       Number of random convolutional kernels.
    normalise : bool, default True
       Whether or not to normalise the input time series per instance.
    n_jobs : int, default=1
       The number of jobs to run in parallel for `transform`. ``-1`` means use all
       processors.
    random_state : None or int, optional, default = None

    See Also
    --------
    MultiRocketMultivariate, MiniRocket, MiniRocketMultivariate

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
    >>> from aeon.transformations.panel.rocket import Rocket
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> trf = Rocket(num_kernels=512)
    >>> trf.fit(X_train)
    Rocket(num_kernels=512)
    >>> X_train = trf.transform(X_train)
    >>> X_test = trf.transform(X_test)
    """

    _tags = {
        "univariate-only": False,
        "fit_is_empty": False,
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
    }

    def __init__(self, num_kernels=10_000, normalise=True, n_jobs=1, random_state=None):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state
        super(Rocket, self).__init__(_output_convert=False)

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
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

        _, n_channels, n_timepoints = X.shape
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, n_channels, self._random_state
        )
        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray [n_instances, num_kernels], transformed features
        """
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        X_ = _apply_kernels(X.astype(np.float32), self.kernels)
        set_num_threads(prev_threads)
        return X_


@njit(
    "Tuple((float32[:],int32[:],float32[:],int32[:],int32[:],int32[:],"
    "int32[:]))(int32,int32,int32,optional(int32))",
    cache=True,
)
def _generate_kernels(n_timepoints, num_kernels, n_channels, seed):
    if seed is not None:
        np.random.seed(seed)
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels).astype(np.int32)

    num_channel_indices = np.zeros(num_kernels, dtype=np.int32)
    for i in range(num_kernels):
        limit = min(n_channels, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(num_kernels):
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


@njit(fastmath=True, cache=True)
def _apply_kernel_univariate(X, weights, length, bias, dilation, padding):
    n_timepoints = len(X)

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

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
    n_columns, n_timepoints = X.shape

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

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


@njit(
    "float32[:,:](float32[:,:,:],Tuple((float32[::1],int32[:],float32[:],"
    "int32[:],int32[:],int32[:],int32[:])))",
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
        num_channel_indices,
        channel_indices,
    ) = kernels

    n_instances, n_channels, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros(
        (n_instances, num_kernels * 2), dtype=np.float32
    )  # 2 features per kernel

    for i in prange(n_instances):
        a1 = 0  # for weights
        a2 = 0  # for channel_indices
        a3 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 2

            if num_channel_indices[j] == 1:
                _X[i, a3:b3] = _apply_kernel_univariate(
                    X[i, channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )

            else:
                _weights = weights[a1:b1].reshape((num_channel_indices[j], lengths[j]))

                _X[i, a3:b3] = _apply_kernel_multivariate(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2],
                )

            a1 = b1
            a2 = b2
            a3 = b3

    return _X.astype(np.float32)
