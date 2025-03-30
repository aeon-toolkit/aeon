"""EIF(Extended Isolation Forest) anomaly detector."""

__maintainer__ = ["Akhil-Jasson"]
__all__ = ["EIF"]

import numpy as np
from sklearn.preprocessing import StandardScaler

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.utils.validation._dependencies import _check_soft_dependencies


class EIF(BaseAnomalyDetector):
    """Extended Isolation Forest (EIF) for anomaly detection using h2o.ai.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of isolation trees in the ensemble.
    sample_size : float or int, default='auto'
        The number of samples to draw from X to train each base estimator.
        - If float, should be between 0.0 and 1.0 and represents the proportion
          of the dataset to draw for training each base estimator.
        - If int, represents the absolute number of samples.
        - If 'auto', sample_size is set to min(256, n_samples).
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
    extension_level : int, default=None
        The extension level of the isolation forest. If None, an appropriate value
        will be determined based on the dimensionality of the data.
    random_state : int, RandomState instance, default=None
        Controls the pseudo-randomization process.
    axis : int, default=1
        The time point axis of the input series if it is 2D. If ``axis==0``, it is
        assumed each column is a time series and each row is a time point. i.e. the
        shape of the data is ``(n_timepoints, n_channels)``. ``axis==1`` indicates
        the time series are in rows, i.e. the shape of the data is
        ``(n_channels, n_timepoints)``.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "requires_y": False,
        "X_inner_type": "np.ndarray",
        "python_dependencies": ["h2o"],
    }

    def __init__(
        self,
        n_estimators=100,
        sample_size="auto",
        contamination=0.1,
        extension_level=None,
        random_state=None,
        axis=1,
    ):
        _check_soft_dependencies(self._tags["python_dependencies"])
        super().__init__(axis=axis)

        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.contamination = contamination
        self.extension_level = extension_level
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _fit(self, X, y=None):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_timepoints,) or (n_timepoints, n_channels)
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        from h2o.init import init
        from h2o.is_initialized import is_initialized
        from h2o.frame import H2OFrame
        from h2o.estimators.isolation_forest import H2OExtendedIsolationForestEstimator

        # Initialize h2o if not already initialized
        if not is_initialized():
            init(silent=True)

        # Fit the scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Convert to h2o frame
        train_df = H2OFrame(X_scaled)

        # Set up the model parameters
        sample_size_param = self.sample_size
        if sample_size_param == "auto":
            sample_size_param = min(256, X_scaled.shape[0])

        self.eif = H2OExtendedIsolationForestEstimator(
            ntrees=self.n_estimators,
            sample_size=sample_size_param,
            extension_level=self.extension_level,
            seed=self.random_state,
        )

        # Train the model
        self.eif.train(training_frame=train_df)

        # Calculate threshold based on contamination
        if self.contamination > 0:
            preds = self.eif.predict(train_df)
            anomaly_scores = preds.as_data_frame()["anomaly_score"].values
            self.threshold_ = np.percentile(
                anomaly_scores, 100 * (1 - self.contamination)
            )
        else:
            self.threshold_ = 0

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict anomaly scores for X.

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_timepoints,) or (n_timepoints, n_channels)

        Returns
        -------
        np.ndarray
            The anomaly scores of the input samples.
            The higher, the more abnormal.
        """
        from h2o.frame import H2OFrame

        # Transform the data using the fitted scaler
        X_scaled = self.scaler.transform(X)

        # Convert to h2o frame
        test_df = H2OFrame(X_scaled)

        # Get anomaly scores
        preds = self.eif.predict(test_df)
        scores = preds.as_data_frame()["anomaly_score"].values

        return scores

    def predict_labels(self, X, axis=1) -> np.ndarray:
        """Predict if points are anomalies or not.

        Parameters
        ----------
        X : one of aeon.base._base_series.VALID_SERIES_INPUT_TYPES
            The time series to predict for.
        axis : int, default=1
            The time point axis of the input series if it is 2D.

        Returns
        -------
        np.ndarray
            Returns 1 for anomalies/outliers and 0 for inliers.
        """
        # Use base class to handle input preprocessing
        self._check_is_fitted()
        X = self._preprocess_series(X, axis, False)

        # Get anomaly scores
        scores = self._predict(X)

        # Use threshold to determine outliers
        predictions = np.zeros(len(X), dtype=int)
        predictions[scores > self.threshold_] = 1

        return predictions