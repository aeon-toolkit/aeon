"""Slope transformer."""

__all__ = ["SlopeTransformer"]
__maintainer__ = []

import math

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.split import split_series


class SlopeTransformer(BaseCollectionTransformer):
    """Piecewise slope transformation.

    Class to perform a slope transformation on a collection of time series.
    Numpy array of shape (n_cases, n_channels, n_timepoints) is
    transformed to numpy array of shape (n_cases, n_channels, n_intervals).
    The new feature is the slope over that interval found using a
    total least squares regression (note that total least squares is different
    from ordinary least squares regression.)

    Parameters
    ----------
    n_intervals : int, number of approx equal segments
                    to split the time series into.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection import SlopeTransformer
    >>> X = np.array([[[4, 6, 10, 12, 8, 6, 5, 5]]])
    >>> s = SlopeTransformer(n_intervals=2)
    >>> res = s.fit_transform(X)
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
    }

    def __init__(self, n_intervals=8):
        self.n_intervals = n_intervals
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
        collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        3D np.ndarray of shape (n_cases, n_channels, n_intervals)
        """
        # Get information about the dataframe
        n_cases, n_channels, n_timepoints = X.shape
        self._check_parameters(n_timepoints)
        full_data = []
        for i in range(n_cases):
            case_data = []
            for j in range(n_channels):
                splits = split_series(X[i][j], self.n_intervals)
                # Calculate gradients
                res = [self._get_gradient(x) for x in splits]
                case_data.append(res)
            full_data.append(np.asarray(case_data))

        return np.array(full_data)

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
        # Create an array that contains 1,2,3,...,len(Y) for the x coordinates.
        X = np.arange(1, len(Y) + 1)

        # Calculate the mean of both arrays
        meanX = np.mean(X)
        meanY = np.mean(Y)

        # Calculate (yi-mean(y))^2
        yminYbar = (Y - meanY) ** 2

        # Calculate (xi-mean(x))^2
        xminXbar = (X - meanX) ** 2

        # Sum them to produce w.
        w = np.sum(yminYbar) - np.sum(xminXbar)

        # Sum (xi-mean(x))*(yi-mean(y)) and multiply by 2 to calculate r
        r = 2 * np.sum((X - meanX) * (Y - meanY))

        if r == 0:
            # remove nans
            m = 0
        else:
            # Gradient is defined as (w+sqrt(w^2+r^2))/r
            m = (w + math.sqrt(w**2 + r**2)) / r

        return m

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
