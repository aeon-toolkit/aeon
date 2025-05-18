"""Adapter to use classification algorithms for collection anomaly detection."""

__maintainer__ = []
__all__ = ["ClassificationAdapter"]

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from aeon.anomaly_detection.collection.base import BaseCollectionAnomalyDetector
from aeon.base._base import _clone_estimator
from aeon.classification.feature_based import SummaryClassifier


class ClassificationAdapter(BaseCollectionAnomalyDetector):
    """
    Basic classifier adapter for collection anomaly detection.

    This class wraps a classification algorithm to be used as an anomaly detector.
    Anomaly labels are required for training.

    Parameters
    ----------
    classifier : aeon classifier or ClassifierMixin
        The classification algorithm to be adapted.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    """

    _tags = {
        "fit_is_empty": False,
        "requires_y": True,
        "anomaly_output_type": "binary",
        "learning_type:supervised": True,
    }

    def __init__(self, classifier, random_state=None):
        self.classifier = classifier
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y=None):
        if not isinstance(self.classifier, ClassifierMixin):
            raise ValueError(
                "The estimator must be an aeon classification algorithm "
                "or class that implements the ClassifierMixin interface."
            )

        self.classifier_ = _clone_estimator(
            self.classifier, random_state=self.random_state
        )
        self.classifier_.fit(X, y)
        return self

    def _predict(self, X):
        return self.classifier_.predict(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "classifier": SummaryClassifier(
                estimator=RandomForestClassifier(n_estimators=5)
            )
        }
