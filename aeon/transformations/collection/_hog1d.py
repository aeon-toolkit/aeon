"""HOG1D transform."""

import math
import numbers

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.split import split_series


class HOG1DTransformer(BaseCollectionTransformer):
    """HOG1D transform.

    This transformer calculates the HOG1D transform [1] of a collection of time series.
    HOG1D splits each time series n_intervals times, and finds a histogram of
    gradients within each interval.

    Parameters
    ----------
    n_intervals : int
        Length of interval.
    n_bins : int
        Number of bins in the histogram.
    scaling_factor : float
        A constant that is multiplied to modify the distribution.

    References
    ----------
    [1] J. Zhao and L. Itti "Classifying time series using local descriptors with
    hybrid sampling", IEEE Transactions on Knowledge and Data Engineering 28(3), 2015.


    """

    _tags = {
        "fit_is_empty": True,
    }

    def __init__(self, n_intervals=2, n_bins=8, scaling_factor=0.1):
        self.n_intervals = n_intervals
        self.n_bins = n_bins
        self.scaling_factor = scaling_factor
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, 1, n_timepoints)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        X : 3D np.ndarray of shape = [n_cases, 1, feature_length]
            collection of time series to transform

        """
        # Get information about the dataframe
        n_cases, n_channels, n_timepoints = X.shape
        # Check the parameters are appropriate
        self._check_parameters(n_timepoints)
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
        X : a numpy array of shape = [time_n_timepoints]

        Returns
        -------
        HOG1Ds : a numpy array of shape = [num_intervals*num_bins].
                 It contains the histogram of each gradient within
                 each interval.
        """
        # Firstly, split the time series into approx equal
        # length intervals
        splitTimeSeries = split_series(X, self.n_intervals)
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
        histogram = [0.0] * self.n_bins

        # Calculate the gradients of each element
        for i in range(1, len(X) - 1):
            gradients[(i - 1)] = self.scaling_factor * 0.5 * (X[(i + 1)] - X[(i - 1)])

        # Calculate the orientations
        orients = [math.degrees(math.atan(x)) for x in gradients]

        # Calculate the boundaries of the histogram
        hisBoundaries = [
            -90 + (180 / self.n_bins) + ((180 / self.n_bins) * x)
            for x in range(self.n_bins)
        ]

        # Construct the histogram
        for x in range(len(orients)):
            orientToAdd = orients[x]
            for y in range(len(hisBoundaries)):
                if orientToAdd <= hisBoundaries[y]:
                    histogram[y] += 1.0
                    break

        return histogram

    def _check_parameters(self, n_timepoints):
        """Check the values of parameters inserted into HOG1D.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.n_intervals, int):
            if self.n_intervals <= 0:
                raise ValueError("num_intervals must have the value of at least 1")
            if self.n_intervals > n_timepoints:
                raise ValueError("num_intervals cannot be higher than serie_length")
        else:
            raise TypeError(
                f"num_intervals must be an 'int' Found {type(self.n_intervals)} "
                f"instead."
            )

        if isinstance(self.n_bins, int):
            if self.n_bins <= 0:
                raise ValueError("num_bins must have the value of at least 1")
        else:
            raise TypeError(
                f"num_bins must be an 'int'. Found"
                f" {type(self.n_bins).__name__}instead."
            )

        if not isinstance(self.scaling_factor, numbers.Number):
            raise TypeError(
                f"scaling_factor must be a 'number'. Found"
                f" {type(self.scaling_factor).__name__}instead."
            )
