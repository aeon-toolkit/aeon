"""Differencing transformations."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["DifferencingSeriesTransformer"]

from aeon.transformations.series.base import BaseSeriesTransformer


class DifferencingSeriesTransformer(BaseSeriesTransformer):
    """Differencing transformations.

    This transformer returns the differenced series of the input time series.
    The differenced series is obtained by subtracting the previous value
    from the current value.

    Examples
    --------
    >>> from aeon.transformations.series import DifferencingSeriesTransformer
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = DifferencingSeriesTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def __init__(
        self,
    ):
        super().__init__(axis=1)

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            Data to be transformed, shape (n_channels, n_timepoints)
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        X = X[0]
        return X[1:] - X[:-1]
