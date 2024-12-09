"""ROCKAD anomaly detector."""

__all__ = ["ROCKAD"]

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.utils import resample

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.windowing import reverse_windowing, sliding_windows


class ROCKAD(BaseAnomalyDetector):
    """
    ROCKET-based Anomaly Detector (ROCKAD).

    ROCKAD leverages the ROCKET transformation for feature extraction from
    time series data and applies the scikit learn k-nearest neighbors (k-NN)
    approach with bootstrap aggregation for robust anomaly detection.
    After windowing the data, the data gets transformed into the ROCKET feature space.
    Then the windows are compared based on the feature space by
    finding the nearest neighbours.

    This class supports both univariate and multivariate time series and
    provides options for normalizing features, applying power transformations,
    and customizing the distance metric.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of k-NN estimators to use in the bootstrap aggregation.
    n_kernels : int, default=100
        Number of kernels to use in the ROCKET transformation.
    normalise : bool, default=True
        Whether to normalize the ROCKET-transformed features.
    n_neighbors : int, default=5
        Number of neighbors to use for the k-NN algorithm.
    n_jobs : int, default=1
        Number of parallel jobs to use for the k-NN algorithm and ROCKET transformation.
    metric : str, default="euclidean"
        Distance metric to use for the k-NN algorithm.
    power_transform : bool, default=True
        Whether to apply a power transformation (e.g., Yeo-Johnson) to the features.
    window_size : int, default=10
        Size of the sliding window for segmenting input time series data.
    stride : int, default=1
        Step size for moving the sliding window over the time series data.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    rocket_transformer_ : Optional[Rocket]
        Instance of the ROCKET transformer used to extract features, set after fitting.
    estimator_ : Optional[NearestNeighbors]
        k-NN estimator used for anomaly scoring, set after fitting.
    scaler_ : StandardScaler
        Scaler used for normalizing the ROCKET-transformed features.
    power_transformer_ : PowerTransformer
        Transformer used to apply power transformation to the features.
    n_inf_cols_ : list
        List of feature columns containing infinite values, identified during fitting.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:multithreading": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        n_estimators=10,
        n_kernels=100,
        normalise=True,
        n_neighbors=5,
        n_jobs=1,
        metric="euclidean",
        power_transform=True,
        window_size: int = 10,
        stride: int = 1,
        random_state=42,
    ):
        super().__init__(axis=0)

        self.n_estimators = n_estimators
        self.n_kernels = n_kernels
        self.normalise = normalise
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.metric = metric
        self.power_transform = power_transform
        self.window_size = window_size
        self.stride = stride
        self.random_state = random_state

        self.rocket_transformer_: Optional[Rocket] = None
        self.estimator_: Optional[NearestNeighbors] = None
        self.scaler_: Optional[StandardScaler] = None
        self.power_transformer_: Optional[PowerTransformer] = None
        self.inf_columns_ = None

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ROCKAD":
        self._check_params(X)

        _X, _ = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )

        self._inner_fit(_X)

        return self

    def _check_params(self, X: np.ndarray) -> None:
        if self.window_size < 1 or self.window_size > X.shape[0]:
            raise ValueError(
                "The window size must be at least 1 and at most the length of the "
                "time series."
            )

        if self.stride < 1 or self.stride > self.window_size:
            raise ValueError(
                "The stride must be at least 1 and at most the window size."
            )

    def _inner_fit(self, X: np.ndarray) -> None:

        if X.shape[0] < self.n_neighbors:
            raise ValueError(
                f"Window count ({X.shape[0]}) has to be larger than "
                f"n_neighbors ({self.n_neighbors})."
                "Please choose a smaller n_neighbors value or increase window count "
                "by choosing a smaller window size or larger stride."
            )

        self.rocket_transformer_ = Rocket(
            n_kernels=self.n_kernels,
            normalise=self.normalise,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        Xt = self.rocket_transformer_.fit_transform(X)

        Xt = Xt.astype("float64")
        self.Xtp = None  # X: values, t: (rocket) transformed, p: power transformed

        if self.power_transform is True:
            self.power_transformer_ = PowerTransformer(standardize=False)
            try:
                self.Xtp = self.power_transformer_.fit_transform(Xt)
            except Exception as e:
                warnings.warn(
                    "Power Transform failed and thus has been disabled. "
                    "Consider adjusting window_size and/or stride and retrying. "
                    f"PowerTransformer failed: {e}. ",
                    UserWarning,
                    stacklevel=2,
                )
                self.Xtp = Xt  # Fallback to raw data
                self.power_transform = False
            self.Xtp_df = pd.DataFrame(self.Xtp)

        else:
            self.Xtp_df = pd.DataFrame(Xt)

        Xtp_scaled = None

        if self.power_transform is True:

            inf_columns = self.Xtp_df.columns[np.isinf(self.Xtp_df).any(axis=0)]
            self.Xtp_df = self.Xtp_df.drop(columns=inf_columns)
            self.inf_columns_ = inf_columns
            self.scaler_ = StandardScaler()
            Xtp_scaled = self.scaler_.fit_transform(self.Xtp_df)
        else:
            Xtp_scaled = self.Xtp_df.astype(np.float64)

        self.list_baggers = []

        for idx_estimator in range(self.n_estimators):
            # Initialize estimator
            self.estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                n_jobs=self.n_jobs,
                metric=self.metric,
                algorithm="kd_tree",
            )
            # Bootstrap Aggregation
            Xtp_scaled_sample = resample(
                Xtp_scaled,
                replace=True,
                n_samples=None,
                random_state=self.random_state + idx_estimator,
                stratify=None,
            )

            Xtp_scaled_sample = Xtp_scaled_sample

            # Fit estimator and append to estimator list
            self.estimator.fit(Xtp_scaled_sample)
            self.list_baggers.append(self.estimator)

    def _predict(self, X) -> np.ndarray:

        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )

        point_anomaly_scores = self._inner_predict(_X, padding)

        return point_anomaly_scores

    def _fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_params(X)
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )

        self._inner_fit(_X)
        point_anomaly_scores = self._inner_predict(_X, padding)
        return point_anomaly_scores

    def _inner_predict(self, X: np.ndarray, padding: int) -> np.ndarray:

        anomaly_scores = self._predict_proba(X)

        point_anomaly_scores = reverse_windowing(
            anomaly_scores, self.window_size, np.nanmean, self.stride, padding
        )

        point_anomaly_scores = (point_anomaly_scores - point_anomaly_scores.min()) / (
            point_anomaly_scores.max() - point_anomaly_scores.min()
        )

        return point_anomaly_scores

    def _predict_proba(self, X):
        """
        Predicts the probability of anomalies for the input data.

        Parameters
        ----------
            X (array-like): The input data.

        Returns
        -------
            np.ndarray: The predicted probabilities.

        """
        y_scores = np.zeros((len(X), self.n_estimators))
        # Transform into rocket feature space
        Xt = self.rocket_transformer_.transform(X)
        Xtp_scaled = None

        if self.power_transform is True:
            # Power Transform using yeo-johnson
            Xtp = self.power_transformer_.transform(Xt)
            Xtp_df = pd.DataFrame(Xtp)

            Xtp_df = Xtp_df.drop(columns=self.inf_columns_, errors="ignore")

            Xtp_df = replace_inf_and_nan(Xtp_df)

            Xtp_scaled = self.scaler_.transform(Xtp_df)
            Xtp_scaled = pd.DataFrame(Xtp_scaled, columns=Xtp_df.columns)

            Xtp_scaled = replace_inf_and_nan(Xtp_scaled)

            Xtp_scaled = Xtp_scaled.astype(np.float64).to_numpy()

        else:
            Xtp_scaled = Xt.astype(np.float64)

        for idx, bagger in enumerate(self.list_baggers):
            # Get scores from each estimator
            distances, _ = bagger.kneighbors(Xtp_scaled)

            # Compute mean distance of nearest points in window
            scores = distances.mean(axis=1).reshape(-1, 1)
            scores = scores.squeeze()

            y_scores[:, idx] = scores

        # Average the scores to get the final score for each time series
        y_scores = y_scores.mean(axis=1)

        return y_scores


def replace_inf_and_nan(df, inf_replacement=0.0, nan_replacement=0.0):
    """
    Replace infinite and NaN values in the given DataFrame with specified values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to process.
    inf_replacement : float or int, optional
        The value to replace infinite values with. Default is 0.0.
    nan_replacement : float or int, optional
        The value to replace NaN values with. Default is 0.0.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with no infinite or NaN values.
    """
    # Replace infinite values
    infinite_mask = np.isinf(df)
    if infinite_mask.any().any():
        df[infinite_mask] = inf_replacement

    # Replace NaN values
    if df.isna().any().any():
        df = df.fillna(nan_replacement)

    return df
