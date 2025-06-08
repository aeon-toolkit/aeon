"""Adapter to use outlier detection algorithms for collection anomaly detection."""

__maintainer__ = []
__all__ = ["OutlierDetectionAdapter"]

from sklearn.base import OutlierMixin
from sklearn.ensemble import IsolationForest

from aeon.anomaly_detection.collection.base import BaseCollectionAnomalyDetector
from aeon.base._base import _clone_estimator


class OutlierDetectionAdapter(BaseCollectionAnomalyDetector):
    """
    Basic outlier detection adapter for collection anomaly detection.

    This class wraps an sklearn outlier detection algorithm to be used as an anomaly
    detector.

    Parameters
    ----------
    detector : OutlierMixin
        The outlier detection algorithm to be adapted.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    """

    _tags = {
        "X_inner_type": "numpy2D",
        "fit_is_empty": False,
        "anomaly_output_type": "binary",
        "learning_type:semi_supervised": True,
    }

    def __init__(self, detector, random_state=None):
        self.detector = detector
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y=None):
        if not isinstance(self.detector, OutlierMixin):
            raise ValueError(
                "The estimator must be an outlier detection algorithm "
                "that implements the OutlierMixin interface."
            )

        self.detector_ = _clone_estimator(self.detector, random_state=self.random_state)
        self.detector_.fit(X, y)
        return self

    def _predict(self, X):
        pred = self.detector_.predict(X)
        pred[pred == -1] = 0
        return pred

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"detector": IsolationForest(n_estimators=3)}
