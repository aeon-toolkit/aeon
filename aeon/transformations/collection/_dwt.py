"""Discrete wavelet transform."""

__maintainer__ = []

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.numba.wavelets import haar_transform


class DWTTransformer(BaseCollectionTransformer):
    """Discrete Wavelet Transform Transformer.

    Performs the Haar wavelet transformation on a time series.

    Parameters
    ----------
    n_levels : int, number of levels to perform the Haar wavelet
                 transformation.

    Examples
    --------
    >>> from aeon.transformations.collection import DWTTransformer
    >>> import numpy as np
    >>> data = np.array([[[1,2,3,4,5,6,7,8,9,10]],[[5,5,5,5,5,5,5,5,5,5]]])
    >>> dwt = DWTTransformer(n_levels=2)
    >>> data2 = dwt.fit_transform(data)

    """

    _tags = {
        "fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(self, n_levels=3):
        self.n_levels = n_levels
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            collection of transformed time series
        """
        _X = np.array(X, dtype=np.float64)
        n_cases, n_channels, n_timepoints = _X.shape
        _X = np.swapaxes(_X, 0, 1)
        self._check_parameters()

        # On each dimension, perform PAA
        channels = []
        for i in range(n_channels):
            channels.append(self._extract_wavelet_coefficients(_X[i]))
        result = np.array(channels)
        result = np.swapaxes(result, 0, 1)

        return result

    def _extract_wavelet_coefficients(self, data):
        """Extract wavelet coefficients of a 2d array of time series.

        The coefficients correspond to the wavelet coefficients
        from levels 1 to num_levels followed by the approximation
        coefficients of the highest level.
        """
        num_levels = self.n_levels
        res = []

        for x in data:
            if num_levels == 0:
                res.append(x)
            else:
                coeffs = []
                current = x
                approx = None
                for _ in range(num_levels):
                    approx, wav_coeffs = haar_transform(current)
                    current = approx
                    coeffs.extend(wav_coeffs[::-1])
                coeffs.extend(approx[::-1])
                coeffs.reverse()
                res.append(coeffs)

        return res

    def _check_parameters(self):
        """Check the values of parameters passed to DWT.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.n_levels, int):
            if self.n_levels <= -1:
                raise ValueError("num_levels must have the value" + "of at least 0")
        else:
            raise TypeError(
                "num_levels must be an 'int'. Found"
                + "'"
                + type(self.n_levels).__name__
                + "' instead."
            )
