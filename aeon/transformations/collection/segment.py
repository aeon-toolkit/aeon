# -*- coding: utf-8 -*-
"""Interval and window segmenter transformers."""

import math

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


class SlidingWindowSegmenter(BaseCollectionTransformer):
    """Sliding window segmenter transformer.

    This class is to transform a univariate series into a multivariate one by
    extracting sets of subsequences. It does this by firstly padding the time series
    on either end floor(window_length/2) times. Then it performs a sliding
    window of size window_length and hop size 1.

    e.g. if window_length = 3

    S = 1,2,3,4,5, floor(3/2) = 1 so S would be padded as

    1,1,2,3,4,5,5

    then SlidingWindowSegmenter would extract the following:

    (1,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,5)

    the time series is now a multivariate one.

    Proposed in the ShapeDTW algorithm.

    Parameters
    ----------
        window_length : int, optional, default=5.
            length of sliding window interval

    Returns
    -------
        np.array [n_instances, n_timepoints, window_length]

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.transformations.collection.segment import SlidingWindowSegmenter
    >>> data = np.array([[[1, 2, 3, 4, 5, 6, 7, 8]], [[5, 5, 5, 5, 5, 5, 5, 5]]])
    >>> seggy = SlidingWindowSegmenter(window_length=4)
    >>> data2 = seggy.fit_transform(data)
    """

    _tags = {
        "univariate-only": True,
        "fit_is_empty": True,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
    }

    def __init__(self, window_length=5):
        self.window_length = window_length
        super(SlidingWindowSegmenter, self).__init__()

    def _transform(self, X, y=None):
        """Transform time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, 1, series_length)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        X : 3D np.ndarray of shape = (n_cases, series_length, window_length)
            windowed series
        """
        # get the number of attributes and instances
        if X.shape[1] > 1:
            raise ValueError("Segmenter does not support multivariate")
        X = X.squeeze(1)

        n_timepoints = X.shape[1]
        n_instances = X.shape[0]

        # Check the parameters are appropriate
        self._check_parameters(n_timepoints)

        pad_amnt = math.floor(self.window_length / 2)
        padded_data = np.zeros((n_instances, n_timepoints + (2 * pad_amnt)))

        # Pad both ends of X
        for i in range(n_instances):
            padded_data[i] = np.pad(X[i], pad_amnt, mode="edge")

        subsequences = np.zeros((n_instances, n_timepoints, self.window_length))

        # Extract subsequences
        for i in range(n_instances):
            subsequences[i] = self._extract_subsequences(padded_data[i], n_timepoints)
        return np.array(subsequences)

    def _extract_subsequences(self, instance, n_timepoints):
        """Extract a set of subsequences from a list of instances.

        Adopted from -
        https://stackoverflow.com/questions/4923617/efficient-numpy-2d-array-
        construction-from-1d-array/4924433#4924433

        """
        shape = (n_timepoints, self.window_length)
        strides = (instance.itemsize, instance.itemsize)
        return np.lib.stride_tricks.as_strided(instance, shape=shape, strides=strides)

    def _check_parameters(self, n_timepoints):
        """Check the values of parameters for interval segmenter.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.window_length, int):
            if self.window_length <= 0:
                raise ValueError(
                    "window_length must have the \
                                  value of at least 1"
                )
        else:
            raise TypeError(
                "window_length must be an 'int'. \
                            Found '"
                + type(self.window_length).__name__
                + "' instead."
            )
