# -*- coding: utf-8 -*-
"""Piecewise Aggregate Approximation Transformer (PAA)."""
import numpy as np

from aeon.transformations.base import BaseTransformer

__author__ = "MatthewMiddlehurst"


class PAA(BaseTransformer):
    """Piecewise Aggregate Approximation Transformer (PAA).

    (PAA) Piecewise Aggregate Approximation Transformer, as described in [1]. For
    each series reduce the dimensionality to num_intervals, where each value is the
    mean of values in the interval.

    Parameters
    ----------
    n_intervals   : int, dimension of the transformed data (default 8)

    Notes
    -----
    [1]  Eamonn Keogh, Kaushik Chakrabarti, Michael Pazzani, and Sharad Mehrotra.
    Dimensionality reduction for fast similarity search in large time series
    databases. Knowledge and information Systems, 3(3), 263-286, 2001.

    Examples
    --------
    >>> from aeon.transformations.panel.dictionary_based import PAA
    >>> import numpy as np
    >>> data = np.array([[[1,2,3,4,5,6,7,8,9,10]],[[5,5,5,5,5,5,5,5,5,5]]])
    >>> paa = PAA(n_intervals=2)
    >>> data2 = paa.fit_transform(data)
    """

    _tags = {
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
    }

    def __init__(self, n_intervals=8):
        self.n_intervals = n_intervals
        super(PAA, self).__init__()

    def set_num_intervals(self, n):
        """Set self.num_intervals to n."""
        self.n_intervals = n

    def _transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
            collection of transformed time series
        """
        # Get information about the dataframe
        # PAA acts on each dimension, so its best to reshape prior to transform
        n_cases, n_channels, series_length = X.shape
        _X = np.swapaxes(X, 0, 1)

        # Check the parameters are appropriate
        self._check_parameters(series_length)

        # On each dimension, perform PAA
        channels = []
        for i in range(n_channels):
            channels.append(self._perform_paa_along_dim(_X[i]))
        result = np.array(channels)
        result = np.swapaxes(result, 0, 1)

        return result

    def _perform_paa_along_dim(self, X):
        n_cases, series_length = X.shape
        data = []

        for i in range(n_cases):
            series = X[i, :]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = series_length / self.n_intervals
            frame_sum = 0

            for n in range(series_length):
                remaining = frame_length - current_frame_size

                if remaining > 1:
                    frame_sum += series[n]
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    current_frame_size += remaining

                if current_frame_size == frame_length:
                    frames.append(frame_sum / frame_length)
                    current_frame += 1

                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining

            # if the last frame was lost due to double imprecision
            if current_frame == self.n_intervals - 1:
                frames.append(frame_sum / frame_length)

            data.append(frames)

        return data

    def _check_parameters(self, series_length):
        """Check parameters of PAA.

        Function for checking the values of parameters inserted into PAA.
        For example, the number of subsequences cannot be larger than the
        time series length.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.n_intervals, int):
            if self.n_intervals <= 0:
                raise ValueError("num_intervals must have the value of at least 1")
            if self.n_intervals > series_length:
                raise ValueError(
                    "num_intervals cannot be higher than the time series length."
                )
        else:
            raise TypeError(
                f"num_intervals must be an 'int'. Found"
                f" {type(self.n_intervals).__name__} instead."
            )
