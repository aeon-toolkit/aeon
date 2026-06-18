"""Abstract base class for time series anomaly detectors."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseAnomalyDetector"]

from abc import abstractmethod

import numpy as np
import pandas as pd

from aeon.base import BaseAeonEstimator
from aeon.utils.validation.labels import check_anomaly_detection_y


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

    def _check_y(self, y, correct_length) -> np.ndarray:
        """Check y input is valid.

        Must be 1-dimensional and contain only 0s (no anomaly) and 1s (anomaly).
        """
        # Remind user if y is not required for this estimator on failure
        req_msg = (
            f" {self.__class__.__name__} does not require a y input."
            if self.get_tag("requires_y")
            else ""
        )

        if isinstance(y, pd.DataFrame):
            # only accept size 1 dataframe
            if y.shape[1] > 1:
                raise TypeError(
                    "Error in input type for y: y input as pd.DataFrame should have a "
                    "single column series." + req_msg
                )
            y = y.squeeze().values

        try:
            check_anomaly_detection_y(y)
        except (TypeError, ValueError) as e:
            raise type(e)(f"{e}{req_msg}") from e

        # Check matching number of labels
        n_labels = y.shape[0]
        if n_labels != correct_length:
            raise ValueError(
                f"Mismatch in number of labels. Found {n_labels} and expected "
                f"{correct_length}." + req_msg
            )

        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)
        return y.astype(bool)
