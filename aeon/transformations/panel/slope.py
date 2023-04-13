# -*- coding: utf-8 -*-
"""Slope transformer."""
import math
import statistics

import numpy as np

from aeon.transformations.base import BaseTransformer

__all__ = ["SlopeTransformer"]
__author__ = ["mloning"]


class SlopeTransformer(BaseTransformer):
    """Piecewise slope transformation.

    Class to perform a slope transformation on a collection of time series. Numpy
    array of shape numpy array of shape (n_instances, n_channels, series_length) is
    transformed to numpy array of shape (n_instances, n_channels, n_intervals). The
    new feature is the slope over that interval found using a least squares regression.

    Parameters
    ----------
    n_intervals : int, number of approx equal segments
                    to split the time series into.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.panel.slope import SlopeTransformer
    >>> X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]])
    >>> s = SlopeTransformer(n_intervals=2)
    >>> res = s.fit_transform(X)
    """

    _tags = {
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(self, n_intervals=8):
        self.n_intervals = n_intervals
        super(SlopeTransformer, self).__init__(_output_convert=False)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
        collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        3D np.ndarray of shape = [n_instances, n_channels, series_length] collection
        of time series to transform
        """
        # Get information about the dataframe
        n_cases, n_channels, series_length = X.shape
        self._check_parameters(series_length)
        full_data = []
        for i in range(n_cases):
            case_data = []
            for j in range(n_channels):
                # Calculate gradients
                res = self._get_gradients_of_lines(X[i][j])
                case_data.append(res)
            full_data.append(np.asarray(case_data))

        return np.array(full_data)

    def _get_gradients_of_lines(self, X):
        """Get gradients of lines.

        Function to get the gradients of the line of best fits
        given a time series.

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length]

        Returns
        -------
        gradients : a numpy array of shape = [self.n_intervals].
                    It contains the gradients of the line of best fit
                    for each interval in a time series.
        """
        # Firstly, split the time series into approx equal length intervals
        splitTimeSeries = self._split_time_series(X)
        gradients = []

        for x in range(len(splitTimeSeries)):
            gradients.append(self._get_gradient(splitTimeSeries[x]))

        return gradients

    def _get_gradient(self, Y):
        """Get gradient of lines.

        Function to get the gradient of the line of best fit given a
        section of a time series.

        Equation adopted from:
        real-statistics.com/regression/total-least-squares

        Parameters
        ----------
        Y : a numpy array of shape = [interval_size]

        Returns
        -------
        m : an int corresponding to the gradient of the best fit line.
        """
        # Create a list that contains 1,2,3,4,...,len(Y) for the x coordinates.
        X = [(i + 1) for i in range(len(Y))]

        # Calculate the mean of both lists
        meanX = statistics.mean(X)
        meanY = statistics.mean(Y)

        # Calculate the list (yi-mean(y))^2
        yminYbar = [(y - meanY) ** 2 for y in Y]
        # Calculate the list (xi-mean(x))^2
        xminXbar = [(x - meanX) ** 2 for x in X]

        # Sum them to produce w.
        w = sum(yminYbar) - sum(xminXbar)

        # Calculate the list (xi-mean(x))*(yi-mean(y))
        temp = []
        for x in range(len(X)):
            temp.append((X[x] - meanX) * (Y[x] - meanY))

        # Sum it and multiply by 2 to calculate r
        r = 2 * sum(temp)

        if r == 0:
            # remove nans
            m = 0
        else:
            # Gradient is defined as (w+sqrt(w^2+r^2))/r
            m = (w + math.sqrt(w**2 + r**2)) / r

        return m

    def _split_time_series(self, X):
        """Split a time series into approximately equal intervals.

        Adopted from = https://stackoverflow.com/questions/2130016/
                       splitting-a-list-into-n-parts-of-approximately
                       -equal-length

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length]

        Returns
        -------
        output : a numpy array of shape = [self.n_intervals,interval_size]
        """
        avg = len(X) / float(self.n_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning) : int(beginning + avg)])
            beginning += avg

        return output

    def _check_parameters(self, n_timepoints):
        """Check values of parameters for Slope transformer.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.n_intervals, int):
            if self.n_intervals <= 0:
                raise ValueError(
                    "num_intervals must have the value \
                                  of at least 1"
                )
            if self.n_intervals > n_timepoints:
                raise ValueError(
                    "num_intervals cannot be higher than \
                                  subsequence_length"
                )
        else:
            raise TypeError(
                "num_intervals must be an 'int'. Found '"
                + type(self.n_intervals).__name__
                + "'instead."
            )
