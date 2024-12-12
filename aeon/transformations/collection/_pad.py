"""Padding transformer, pad unequal length time series to max length or fixed length."""

__all__ = ["Padder"]
__maintainer__ = []

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


def _get_max_length(X):
    max_length = X[0].shape[1]
    for x in X:
        if x.shape[1] > max_length:
            max_length = x.shape[1]

    return max_length


class Padder(BaseCollectionTransformer):
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

    fill_value : Union[int, str, np.ndarray], default = 0
        Value to pad with. Can be a float or a statistic string or an numpy array for
        each time series. Supported statistic strings are "mean", "median", "max",
        "min".

    Examples
    --------
    >>> from aeon.transformations.collection import Padder
    >>> import numpy as np
    >>> X = []
    >>> for i in range(10): X.append(np.random.random((4, 75 + i)))
    >>> padder = Padder(pad_length=200, fill_value =42)
    >>> X2 = padder.fit_transform(X)
    >>> X2.shape
    (10, 4, 200)
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "fit_is_empty": False,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "removes_unequal_length": True,
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
        self.fill_value_ = self.fill_value
        max_length = _get_max_length(X)
        if self.pad_length is None:
            self.pad_length_ = max_length
        else:
            if self.pad_length < max_length:
                self.pad_length_ = max_length
            else:
                self.pad_length_ = self.pad_length

        if isinstance(self.fill_value, str):
            if self.fill_value == "mean":
                self.fill_value_ = np.zeros((len(X), X[0].shape[0]))
                for i, series in enumerate(X):
                    for j, channel in enumerate(series):
                        self.fill_value_[i][j] = np.mean(channel)
            elif self.fill_value == "median":
                self.fill_value_ = np.zeros((len(X), X[0].shape[0]))
                for i, series in enumerate(X):
                    for j, channel in enumerate(series):
                        self.fill_value_[i][j] = np.median(channel)
            elif self.fill_value == "min":
                self.fill_value_ = np.zeros((len(X), X[0].shape[0]))
                for i, series in enumerate(X):
                    for j, channel in enumerate(series):
                        self.fill_value_[i][j] = np.min(channel)
            elif self.fill_value == "max":
                self.fill_value_ = np.zeros((len(X), X[0].shape[0]))
                for i, series in enumerate(X):
                    for j, channel in enumerate(series):
                        self.fill_value_[i][j] = np.max(channel)
            else:
                raise ValueError(
                    "Supported modes are mean, median, min, max. \
                                    Please check arguments passed."
                )
        elif isinstance(self.fill_value, np.ndarray):
            if not (len(self.fill_value) == len(X)):
                raise ValueError(
                    "The length of fill_value must match the \
                                length of X if a numpy array is passed as fill_value."
                )
            if not self.fill_value.ndim == 2:
                raise ValueError(
                    """The fill_value argument must be
                                a 2D Numpy array, containing values for
                                each `n_channel` for `n_cases` series."""
                )
        else:
            self.fill_value_ = self.fill_value * np.ones((len(X), X[0].shape[0]))

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
                "max_length of series in transform is greater than the one found in "
                "fit or set in the constructor."
            )
        # Calculate padding amounts

        Xt = []
        for i, series in enumerate(X):
            pad_width = (0, self.pad_length_ - series.shape[1])
            temp_array = []
            for j, channel in enumerate(series):
                # Pad the input array
                padded_array = np.pad(
                    channel,
                    pad_width,
                    mode="constant",
                    constant_values=self.fill_value_[i][j],
                )
                temp_array.append(padded_array)
            Xt.append(temp_array)
        Xt = np.array(Xt)

        return Xt
