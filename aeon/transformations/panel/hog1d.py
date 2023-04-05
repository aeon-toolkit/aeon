# -*- coding: utf-8 -*-
"""HOG1D transform."""
import math
import numbers

import numpy as np

from aeon.transformations.base import BaseTransformer


class HOG1DTransformer(BaseTransformer):
    """HOG1D transform.

    This transformer calculates the HOG1D transform [1] of a collection of time seriess.
    HOG1D splits each time series num_intervals times, and finds a histogram of
    gradients within each interval.

    Parameters
    ----------
        num_intervals   : int, length of interval.
        num_bins        : int, num bins in the histogram.
        scaling_factor  : float, a constant that is multiplied
                          to modify the distribution.

    Notes
    -----
    [1] J. Zhao and L. Itti "Classifying time series using local descriptors with
    hybrid sampling", IEEE Transactions on Knowledge and Data Engineering 28(3), 2015.
    """

    _tags = {
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(self, num_intervals=2, num_bins=8, scaling_factor=0.1):
        self.num_intervals = num_intervals
        self.num_bins = num_bins
        self.scaling_factor = scaling_factor
        super(HOG1DTransformer, self).__init__(_output_convert=False)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        X : 3D np.ndarray of shape = [n_instances, n_dimensions, feature_length]
            collection of time series to transform

        """
        # Get information about the dataframe
        n_cases, n_channels, series_length = X.shape
        if n_channels > 1:
            raise ValueError("HOG1D does not support multivariate time series.")
        # Check the parameters are appropriate
        self._check_parameters(series_length)
        transX = []
        for i in range(n_cases):
            # Get the HOG1Ds of each time series
            inst = self._calculate_hog1ds(X[i][0])
            transX.append(inst)
            # Convert to numpy array
        transX = np.asarray(transX)
        transX = np.reshape(transX, [transX.shape[0], 1, transX.shape[1]])
        return transX

    def _calculate_hog1ds(self, X):
        """Calculate the HOG1Ds given a time series.

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length]

        Returns
        -------
        HOG1Ds : a numpy array of shape = [num_intervals*num_bins].
                 It contains the histogram of each gradient within
                 each interval.
        """
        # Firstly, split the time series into approx equal
        # length intervals
        splitTimeSeries = self._split_time_series(X)
        HOG1Ds = []

        for x in range(len(splitTimeSeries)):
            HOG1Ds.extend(self._get_hog1d(splitTimeSeries[x]))

        return HOG1Ds

    def _get_hog1d(self, X):
        """Get the HOG1D given a portion of a time series.

        X : a numpy array of shape = [interval_size]

        Returns
        -------
        histogram : a numpy array of shape = [num_bins].
        """
        # First step is to pad the portion on both ends once.
        gradients = [0.0] * (len(X))
        X = np.pad(X, 1, mode="edge")
        histogram = [0.0] * self.num_bins

        # Calculate the gradients of each element
        for i in range(1, len(X) - 1):
            gradients[(i - 1)] = self.scaling_factor * 0.5 * (X[(i + 1)] - X[(i - 1)])

        # Calculate the orientations
        orients = [math.degrees(math.atan(x)) for x in gradients]

        # Calculate the boundaries of the histogram
        hisBoundaries = [
            -90 + (180 / self.num_bins) + ((180 / self.num_bins) * x)
            for x in range(self.num_bins)
        ]

        # Construct the histogram
        for x in range(len(orients)):
            orientToAdd = orients[x]
            for y in range(len(hisBoundaries)):
                if orientToAdd <= hisBoundaries[y]:
                    histogram[y] += 1.0
                    break

        return histogram

    def _split_time_series(self, X):
        """Split a time series into approximately equal intervals.

        Adopted from = https://stackoverflow.com/questions/2130016/splitting
                       -a-list-into-n-parts-of-approximately-equal-length

        Parameters
        ----------
        X : a numpy array corresponding to the time series being split
            into approx equal length intervals of shape
            [num_intervals,interval_length].
        """
        avg = len(X) / float(self.num_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning) : int(beginning + avg)])
            beginning += avg

        return output

    def _check_parameters(self, series_length):
        """Check the values of parameters inserted into HOG1D.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError("num_intervals must have the value of at least 1")
            if self.num_intervals > series_length:
                raise ValueError("num_intervals cannot be higher than serie_length")
        else:
            raise TypeError(
                f"num_intervals must be an 'int' Found {type(self.num_intervals)} "
                f"instead."
            )

        if isinstance(self.num_bins, int):
            if self.num_bins <= 0:
                raise ValueError("num_bins must have the value of at least 1")
        else:
            raise TypeError(
                f"num_bins must be an 'int'. Found"
                f" {type(self.num_bins).__name__}instead."
            )

        if not isinstance(self.scaling_factor, numbers.Number):
            raise TypeError(
                f"scaling_factor must be a 'number'. Found"
                f" {type(self.scaling_factor).__name__}instead."
            )
