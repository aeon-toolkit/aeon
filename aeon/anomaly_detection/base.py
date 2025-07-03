"""Abstract base class for time series anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseAnomalyDetector"]

from abc import abstractmethod

import numpy as np

from aeon.base import BaseAeonEstimator


class BaseAnomalyDetector(BaseAeonEstimator):
    """Anomaly detection base class."""

    _tags = {
        "fit_is_empty": True,
        "requires_y": False,
        "learning_type:unsupervised": False,
        "learning_type:semi_supervised": False,
        "learning_type:supervised": False,
    }

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y=None):
        """Fit anomaly detector to X, optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit anomaly detector to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)
        y : Series, default=None
            Additional data, e.g., labels for anomaly detector.

        Returns
        -------
        BaseAnomalyDetector
            The fitted estimator, reference to self.
        """
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Find anomalies in X.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit anomaly detector to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)

        Returns
        -------
        np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence/case is anomalous or its anomaly
            score.
        """
        ...

    @abstractmethod
    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit time series anomaly detector and find anomalies for X.

        Parameters
        ----------
        X : Series or Collection, any supported type
            Data to fit anomaly detector to, of python type as follows:
                Series: 2D np.ndarray shape (n_channels, n_timepoints)
                Collection: 3D np.ndarray shape (n_cases, n_channels, n_timepoints)
                or list of 2D np.ndarray, case i has shape (n_channels, n_timepoints_i)

        Returns
        -------
        np.ndarray
            A boolean, int or float array of length len(X), where each element indicates
            whether the corresponding subsequence/case is anomalous or its anomaly
            score.
        """
        ...
