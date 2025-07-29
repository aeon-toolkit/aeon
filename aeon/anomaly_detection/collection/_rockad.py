"""ROCKAD anomaly detector."""

__all__ = ["ROCKAD"]

import warnings
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample

from aeon.anomaly_detection.collection.base import BaseCollectionAnomalyDetector
from aeon.transformations.collection.convolution_based import Rocket


class ROCKAD(BaseCollectionAnomalyDetector):
    """
    ROCKET-based whole-series Anomaly Detector (ROCKAD).

    ROCKAD [1]_ leverages the ROCKET transformation for feature extraction from
    time series data and applies the scikit learn k-nearest neighbors (k-NN)
    approach with bootstrap aggregation for robust semi-supervised anomaly detection.
    The data gets transformed into the ROCKET feature space.
    Then the whole-series are compared based on the feature space by
    finding the nearest neighbours. The time-point based ROCKAD anomaly detector
    can be found at aeon/anomaly_detection/series/distance_based/_rockad.py

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
    >>> from aeon.anomaly_detection.collection import ROCKAD
    >>> rng = np.random.default_rng(seed=42)
    >>> X_train = rng.normal(loc=0.0, scale=1.0, size=(10, 100))
    >>> X_test = rng.normal(loc=0.0, scale=1.0, size=(5, 100))
    >>> X_test[4][50:58] -= 5
    >>> detector = ROCKAD() # doctest: +SKIP
    >>> detector.fit(X_train) # doctest: +SKIP
    >>> detector.predict(X_test) # doctest: +SKIP
    array([24.11974147, 23.93866453, 21.3941765 , 22.26811959, 64.9630108 ])
    """

    _tags = {
        "anomaly_output_type": "anomaly_scores",
        "learning_type:semi_supervised": True,
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
        normalise=False,
        n_neighbors=5,
        metric="euclidean",
        power_transform=True,
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
        self.random_state = random_state

        self.rocket_transformer_: Optional[Rocket] = None
        self.list_baggers_: Optional[list[NearestNeighbors]] = None
        self.power_transformer_: Optional[PowerTransformer] = None

        super().__init__()

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ROCKAD":
        _X = X
        self._inner_fit(_X)

        return self

    def _inner_fit(self, X: np.ndarray) -> None:

        self.rocket_transformer_ = Rocket(
            n_kernels=self.n_kernels,
            normalise=self.normalise,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        # XT: (n_cases, n_kernels*2)
        Xt = self.rocket_transformer_.fit_transform(X)
        Xt = Xt.astype(np.float64)

        if self.power_transform:
            self.power_transformer_ = PowerTransformer()
            try:
                Xtp = self.power_transformer_.fit_transform(Xt)

            except Exception:
                warnings.warn(
                    "Power Transform failed and thus has been disabled. ",
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
                n_jobs=self.n_jobs,
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
        _X = X
        collection_anomaly_scores = self._inner_predict(_X)

        return collection_anomaly_scores

    def _inner_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the anomaly scores for the input data.

        Parameters
        ----------
            X (array-like): The input data.

        Returns
        -------
            np.ndarray: The predicted probabilities.

        """
        y_scores = np.zeros((len(X), self.n_estimators))
        # Transform into rocket feature space
        # XT: (n_cases, n_kernels*2)
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

        # Average the scores to get the final score for each whole-series
        collection_anomaly_scores = y_scores.mean(axis=1)

        return collection_anomaly_scores
