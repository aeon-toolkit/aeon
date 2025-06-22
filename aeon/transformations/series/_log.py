"""Log transformation."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["LogTransformer"]

import numpy as np

from aeon.transformations.series.base import BaseSeriesTransformer


class LogTransformer(BaseSeriesTransformer):
    """Natural logarithm transformation.

    The Natural logarithm transformation can be used to make the data more normally
    distributed and stabilize its variance.

    Transforms each data point x to log(scale *(x+offset))

    Parameters
    ----------
    offset : float , default = 0
             Additive constant applied to all the data.
    scale  : float , default = 1
             Multiplicative scaling constant applied to all the data.


    Notes
    -----
    The log transformation is applied as :math:`ln(y)`.

    Examples
    --------
    >>> from aeon.transformations.series._log import LogTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = LogTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:inverse_transform": True,
    }

    def __init__(self, offset=0, scale=1):
        self.offset = offset
        self.scale = scale
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        offset = self.offset
        scale = self.scale
        Xt = np.log(scale * (X + offset))
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        Xt = (np.exp(X) / self.scale) - self.offset
        return Xt
