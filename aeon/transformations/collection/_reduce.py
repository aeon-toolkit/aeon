"""Tabularizer transform, for pipelining."""

__maintainer__ = []
__all__ = ["Tabularizer"]

from aeon.transformations.collection.base import BaseCollectionTransformer


class Tabularizer(BaseCollectionTransformer):
    """
    A transformer that turns time series collection into tabular data.

    This estimator converts a 3D numpy into a 2D numpy by concatenating channels
    using ``reshape``. This is only usable with equal length series. This is useful for
    transforming time-series collections into a format that is accepted by sklearn.
    """

    _tags = {
        "fit_is_empty": True,
        "output_data_type": "Tabular",
        "X_inner_type": ["numpy3D"],
        "capability:multivariate": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        """Transform collection into tabular dataframe.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform.
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pandas DataFrame
            Transformed dataframe with only primitives in cells.
        """
        Xt = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        return Xt
