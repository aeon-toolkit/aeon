"""Abstract base class for series time series anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseSeriesAnomalyDetector"]

from abc import abstractmethod
from typing import final

import numpy as np

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.base import BaseSeriesEstimator


class BaseSeriesAnomalyDetector(BaseSeriesEstimator, BaseAnomalyDetector):
    """Base class for series anomaly detection algorithms.

    Anomaly detection algorithms are used to identify anomalous subsequences in time
    series data. These algorithms take a series of length m and return a boolean, int or
    float array of length m, where each element indicates whether the corresponding
    subsequence is anomalous or its anomaly score.

    Input and internal data format (where m is the number of time points and d is the
    number of channels):
        Univariate series (default):
            np.ndarray, shape ``(m,)``, ``(m, 1)`` or ``(1, m)`` depending on axis.
            This is converted to a 2D np.ndarray internally.
            pd.DataFrame, shape ``(m, 1)`` or ``(1, m)`` depending on axis.
            pd.Series, shape ``(m,)``.
        Multivariate series:
            np.ndarray array, shape ``(m, d)`` or ``(d, m)`` depending on axis.
            pd.DataFrame ``(m, d)`` or ``(d, m)`` depending on axis.

    Output data format (one of the following):
        Anomaly scores (default):
            np.ndarray, shape ``(m,)`` of type float. For each point of the input time
            series, the anomaly score is a float value indicating the degree of
            anomalousness. The higher the score, the more anomalous the point.
        Binary classification:
            np.ndarray, shape ``(m,)`` of type bool or int. For each point of the input
            time series, the output is a boolean or integer value indicating whether the
            point is anomalous (``True``/``1``) or not (``False``/``0``).

    Detector learning types:
        Unsupervised (default):
            Unsupervised detectors do not require any training data and can directly be
            used on the target time series. Their tags are set to ``fit_is_empty=True``
            and ``requires_y=False``. You would usually call the ``fit_predict`` method
            on these detectors.
        Semi-supervised:
            Semi-supervised detectors require a training step on a time series without
            anomalies (normal behaving time series). The target value ``y`` would
            consist of only zeros. Thus, these algorithms have logic in the ``fit``
            method, but do not require the target values. Their tags are set to
            ``fit_is_empty=False`` and ``requires_y=False``. You would usually first
            call the ``fit`` method on the training data and then the ``predict``
            method for your target time series.
        Supervised:
            Supervised detectors require a training step on a time series with known
            anomalies (anomalies should be present and must be annotated). The detector
            implements the ``fit`` method, and the target value ``y`` consists of zeros
            and ones. Their tags are, thus, set to ``fit_is_empty=False`` and
            ``requires_y=True``. You would usually first call the ``fit`` method on the
            training data and then the ``predict`` method for your target time series.

    Parameters
    ----------
    axis : int
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.
        Setting this class variable will convert the input data to the chosen axis.
    """

    def __init__(self, axis):
        super().__init__(axis=axis)

    @final
    def fit(self, X, y=None, axis=1):
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
        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self

        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_series(X, axis, True)
        if y is not None:
            y = self._check_y(y, self.metadata_["n_timepoints"])

        self._fit(X=X, y=y)

        # this should happen last
        self.is_fitted = True
        return self

    @final
    def predict(self, X, axis=1) -> np.ndarray:
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
        fit_empty = self.get_tag("fit_is_empty")
        if not fit_empty:
            self._check_is_fitted()

        X = self._preprocess_series(X, axis, False)

        return self._predict(X)

    @final
    def fit_predict(self, X, y=None, axis=1) -> np.ndarray:
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
        if self.get_tag("requires_y"):
            if y is None:
                raise ValueError("Tag requires_y is true, but fit called with y=None")

        # reset estimator at the start of fit
        self.reset()

        X = self._preprocess_series(X, axis, True)

        if self.get_tag("fit_is_empty"):
            self.is_fitted = True
            return self._predict(X)

        if y is not None:
            y = self._check_y(y, self.metadata_["n_timepoints"])

        pred = self._fit_predict(X, y)

        # this should happen last
        self.is_fitted = True
        return pred

    def _fit(self, X, y):
        return self

    @abstractmethod
    def _predict(self, X) -> np.ndarray: ...

    def _fit_predict(self, X, y):
        self._fit(X, y)
        return self._predict(X)
