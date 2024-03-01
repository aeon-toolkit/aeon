"""Implements cosine transformation."""

import numpy as np

from aeon.transformations.base import BaseTransformer

__maintainer__ = []
__all__ = ["CosineTransformer"]


class CosineTransformer(BaseTransformer):
    """Cosine transformation.

    This is a wrapper around numpy's cosine function (see :func:`numpy.cos`).

    See Also
    --------
    numpy.cos

    Examples
    --------
    >>> from aeon.transformations.cos import CosineTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = CosineTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "input_data_type": "Series",
        # what is the abstract type of X: Series, or Panel
        "output_data_type": "Series",
        # what abstract type is returned: Primitives, Series, Panel
        "instancewise": True,
        "X_inner_type": "np.ndarray",
        "y_inner_type": "None",
        "univariate-only": False,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "capability:inverse_transform": False,
    }

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
        Xt = np.cos(X)
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
        Xt = np.arccos(X)
        return Xt
