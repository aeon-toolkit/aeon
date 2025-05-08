"""Abstract base class for time series anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseAnomalyDetector"]

from abc import abstractmethod

import numpy as np

from aeon.base import BaseAeonEstimator


class BaseAnomalyDetector(BaseAeonEstimator):
    """todo base class docs."""

    _tags = {
        # todo
    }

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y=None):
        """Fit time series anomaly detector to X.

        If the tag ``fit_is_empty`` is true, this just sets the ``is_fitted`` tag to
        true. Otherwise, it checks ``self`` can handle ``X``, formats ``X`` into
        the structure required by ``self`` then passes ``X`` (and possibly ``y``) to
        ``_fit``.

        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        y : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES, default=None
            The target values for the time series.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        axis : int
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        BaseSeriesAnomalyDetector
            The fitted estimator, reference to self.
        """
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Find anomalies in X.

        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        axis : int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence is anomalous or its anomaly score.
        """
        ...

    @abstractmethod
    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit time series anomaly detector and find anomalies for X.

        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            The time series to fit the model to.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_INPUT_TYPES for aeon supported types.
        y : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES, default=None
            The target values for the time series.
            A valid aeon time series data structure. See
            aeon.base._base_series.VALID_SERIES_INPUT_TYPES for aeon supported types.
        axis : int, default=1
            The time point axis of the input series if it is 2D. If ``axis==0``, it is
            assumed each column is a time series and each row is a time point. i.e. the
            shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
            the time series are in rows, i.e. the shape of the data is
            ``(n_channels, n_timepoints)``.

        Returns
        -------
        np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence is anomalous or its anomaly score.
        """
        ...
