"""ROCKAD anomaly detector."""

__maintainer__ = []
__all__ = ["ROCKAD"]

import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.transformations.collection.convolution_based import Rocket
from aeon.utils.validation import check_n_jobs
from aeon.utils.windowing import reverse_windowing, sliding_windows


class ROCKAD(BaseSeriesAnomalyDetector):
    """
    ROCKET-based Semi-Supervised Anomaly Detector (ROCKAD).

    Adapted ROCKAD [1]_ version to detect anomalies on time-points.
    ROCKAD leverages the ROCKET transformation for feature extraction from
    time series data and applies the scikit learn k-nearest neighbors (k-NN)
    approach with bootstrap aggregation for robust anomaly detection.
    After windowing, the data gets transformed into the ROCKET feature space.
    Then the windows are compared based on the feature space by
    finding the nearest neighbours. Whole-series based ROCKAD as proposed in
    [1]_ can be found at aeon/anomaly_detection/collection/_rockad.py

    This class supports both univariate and multivariate time series and
    provides options for normalizing features, applying power transformations,
    and customizing the distance metric.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of k-NN estimators to use in the bootstrap aggregation.
    n_kernels : int, default=100
        Number of kernels to use in the ROCKET transformation.
    normalise : bool, default=False
        Whether to normalize the ROCKET-transformed features.
    n_neighbors : int, default=5
        Number of neighbors to use for the k-NN algorithm.
    n_jobs : int, default=1
        Number of parallel jobs to use for the k-NN algorithm and ROCKET transformation.
    metric : str, default="euclidean"
        Distance metric to use for the k-NN algorithm.
    power_transform : bool, default=True
        Whether to apply a power transformation (Yeo-Johnson) to the features.
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
    list_baggers_ : Optional[list[NearestNeighbors]]
        List containing k-NN estimators used for anomaly scoring, set after fitting.
    power_transformer_ : PowerTransformer
        Transformer used to apply power transformation to the features.

    References
    ----------
    .. [1] Theissler, A., Wengert, M., Gerschner, F. (2023).
        ROCKAD: Transferring ROCKET to Whole Time Series Anomaly Detection.
        In: Crémilleux, B., Hess, S., Nijssen, S. (eds) Advances in Intelligent
        Data Analysis XXI. IDA 2023. Lecture Notes in Computer Science,
        vol 13876. Springer, Cham. https://doi.org/10.1007/978-3-031-30047-9_33

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.distance_based import ROCKAD
    >>> rng = np.random.default_rng(seed=42)
    >>> X_train = rng.normal(loc=0.0, scale=1.0, size=(1000,))
    >>> X_test = rng.normal(loc=0.0, scale=1.0, size=(20,))
    >>> X_test[15:20] -= 5
    >>> detector = ROCKAD(window_size=15,n_estimators=10,n_kernels=10,n_neighbors=3)
    >>> detector.fit(X_train)
    ROCKAD(...)
    >>> detector.predict(X_test)
    array([0.        , 0.00554713, 0.0699094 , 0.22881059, 0.32382585,
           0.43652154, 0.43652154, 0.43652154, 0.43652154, 0.43652154,
           0.43652154, 0.43652154, 0.43652154, 0.43652154, 0.43652154,
           0.52382585, 0.65200875, 0.80313368, 0.85194345, 1.        ])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "capability:multithreading": True,
        "fit_is_empty": False,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:semi_supervised": True,
    }

    def __init__(
        self,
        n_estimators=10,
        n_kernels=100,
        normalise=False,
        n_neighbors=5,
        metric="euclidean",
        power_transform=True,
        window_size: int = 10,
        stride: int = 1,
        n_jobs=1,
        random_state=42,
    ):
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

        self.rocket_transformer_: Rocket | None = None
        self.list_baggers_: list[NearestNeighbors] | None = None
        self.power_transformer_: PowerTransformer | None = None

        super().__init__(axis=0)

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ROCKAD":
        self._check_params(X)
        # X: (n_timepoints, 1) because __init__(axis==0)
        _X, _ = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        # _X: (n_windows, window_size)
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

        if int((X.shape[0] - self.window_size) / self.stride + 1) < self.n_neighbors:
            raise ValueError(
                f"Window count ({int((X.shape[0]-self.window_size)/self.stride+1)}) "
                f"has to be larger than n_neighbors ({self.n_neighbors})."
                "Please choose a smaller n_neighbors value or increase window count "
                "by choosing a smaller window size or larger stride."
            )

    def _inner_fit(self, X: np.ndarray) -> None:
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.rocket_transformer_ = Rocket(
            n_kernels=self.n_kernels,
            normalise=self.normalise,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        # X: (n_windows, window_size)
        Xt = self.rocket_transformer_.fit_transform(X)
        # XT: (n_cases, n_kernels*2)
        Xt = Xt.astype(np.float64)

        if self.power_transform:
            self.power_transformer_ = PowerTransformer()
            # todo check if this is still an issue with scikit-learn >= 1.7.0
            # when lower bound is raised
            try:
                Xtp = self.power_transformer_.fit_transform(Xt)

            except Exception:
                warnings.warn(
                    "Power Transform failed and thus has been disabled. "
                    "Try increasing the window size.",
                    UserWarning,
                    stacklevel=2,
                )
                self.power_transformer_ = None
                Xtp = Xt
        else:
            Xtp = Xt

        self.list_baggers_ = []

        for idx_estimator in range(self.n_estimators):
            # Initialize estimator
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                n_jobs=self._n_jobs,
                metric=self.metric,
                algorithm="kd_tree",
            )
            # Bootstrap Aggregation
            Xtp_scaled_sample = resample(
                Xtp,
                replace=True,
                n_samples=None,
                random_state=self.random_state + idx_estimator,
                stratify=None,
            )

            # Fit estimator and append to estimator list
            estimator.fit(Xtp_scaled_sample)
            self.list_baggers_.append(estimator)

    def _predict(self, X) -> np.ndarray:
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )

        point_anomaly_scores = self._inner_predict(_X, padding)

        return point_anomaly_scores

    def _fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self._check_params(X)
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )

        self._inner_fit(_X)
        point_anomaly_scores = self._inner_predict(_X, padding)
        return point_anomaly_scores

    def _inner_predict(self, X: np.ndarray, padding: int) -> np.ndarray:
        """
        Predict the anomaly score for each time-point in the input data.

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

        Xt = Xt.astype(np.float64)

        if self.power_transformer_ is not None:
            # Power Transform using yeo-johnson
            Xtp = self.power_transformer_.transform(Xt)

        else:
            Xtp = Xt

        for idx, bagger in enumerate(self.list_baggers_):
            # Get scores from each estimator
            distances, _ = bagger.kneighbors(Xtp)

            # Compute mean distance of nearest points in window
            scores = distances.mean(axis=1).reshape(-1, 1)
            scores = scores.squeeze()

            y_scores[:, idx] = scores

        # Average the scores to get the final score for each time series
        anomaly_scores = y_scores.mean(axis=1)

        point_anomaly_scores = reverse_windowing(
            anomaly_scores, self.window_size, np.nanmean, self.stride, padding
        )

        point_anomaly_scores = (point_anomaly_scores - point_anomaly_scores.min()) / (
            point_anomaly_scores.max() - point_anomaly_scores.min()
        )

        return point_anomaly_scores
