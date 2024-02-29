"""Padding transformer, pad unequal length time series to max length or fixed length."""

__all__ = ["PaddingTransformer"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


def _get_max_length(X):
    max_length = X[0].shape[1]
    for x in X:
        if x.shape[1] > max_length:
            max_length = x.shape[1]

    return max_length


class PaddingTransformer(BaseCollectionTransformer):
    """Pad unequal length time series to equal, fixed length.

    Pads the input dataset to either fixed length (at least as long as the longest
    series) or finds the max length series across all series and channels and
    pads to that with zeroes.

    Parameters
    ----------
    pad_length  : int or None, default=None
        length to pad the series too. if None, will find the longest sequence and use
        instead. If the pad_length passed is less than the max length, it is reset to
        max length.

    fill_value : int, default = 0
        value to pad with.

    Example
    -------
    >>> from aeon.transformations.collection import PaddingTransformer
    >>> import numpy as np
    >>> X = []
    >>> for i in range(10): X.append(np.random.random((4, 75 + i)))
    >>> padder = PaddingTransformer(pad_length=200, fill_value =42)
    >>> X2 = padder.fit_transform(X)
    >>> X2.shape
    (10, 4, 200)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:unequal_length:removes": True,
    }

    def __init__(self, pad_length=None, fill_value=0):
        self.pad_length = pad_length
        self.fill_value = fill_value
        super().__init__()

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
        max_length = _get_max_length(X)
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
        for series in X:
            pad_width = ((0, 0), (0, self.pad_length_ - series.shape[1]))
            # Pad the input array
            padded_array = np.pad(
                series, pad_width, mode="constant", constant_values=self.fill_value
            )
            Xt.append(padded_array)
        Xt = np.array(Xt)
        return Xt
