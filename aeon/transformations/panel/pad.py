# -*- coding: utf-8 -*-
"""Padding transformer, pad unequal length panel to max length or fixed length."""
import numpy as np

from aeon.transformations.base import BaseTransformer

__all__ = ["PaddingTransformer"]
__author__ = ["abostrom"]


class PaddingTransformer(BaseTransformer):
    """Padding panel of unequal length time series to equal, fixed length.

    Pads the input dataset to either a optional fixed length
    (longer than the longest series).
    Or finds the max length series across all series and dimensions and
    pads to that with zeroes.

    Parameters
    ----------
    pad_length  : int, optional (default=None) length to pad the series too.
                if None, will find the longest sequence and use instead.
    """

    _tags = {
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": ["np-list", "numpy3D"],
        "y_inner_mtype": "None",
        "fit_is_empty": False,
        "capability:unequal_length:removes": True,
    }

    def __init__(self, pad_length=None, fill_value=0):
        self.pad_length = pad_length
        self.fill_value = fill_value
        super(PaddingTransformer, self).__init__()

    @staticmethod
    def _get_max_length(X):
        max_length = X[0].shape[1]
        for x in X:
            if x.shape[1] > max_length:
                max_length = x.shape[1]

        return max_length

    def _fit(self, X, y=None):
        """Fit padding transformer to X and y.

        Calculates the max length in X unless padding length passed as an argument.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series or 3D numpy of equal length
            series
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        max_length = _get_max_length
        if self.pad_length is None:
            self.pad_length_ = max_length
        else:
            if self.pad_length < max_length:
                self.pad_length_ = max_length
            else:
                self.pad_length_ = self.pad_length
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series or 3D numpy of equal length
            series
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : numpy3D array (n_cases, n_channels, self.pad_length_)
            padded time series from X.
        """
        max_length = _get_max_length(X)

        if max_length > self.pad_length_:
            raise ValueError(
                "Error: max_length of series \
                    is greater than the one found when fit or set."
            )
        # Calculate padding amounts
        Xt = []
        for x in X:
            pad_width = ((0, 0), (0, self.pad_length_ - x.shape[1]))
            # Pad the input array
            padded_array = np.pad(
                X, pad_width, mode="constant", constant_values=self.fill_value
            )
            Xt.append(padded_array)
        Xt = np.array(Xt)
        return Xt


def _get_max_length(X):
    def get_length(input):
        return max(map(lambda series: len(series), input))

    return max(map(get_length, X))
