"""Mock forecasters for testing and debugging."""

__author__ = ["ltsaprounis"]

__all__ = [
    "MockForecaster",
    "MockUnivariateForecasterLogger",
    "make_mock_estimator",
    "MockClassifier",
    "MockClassifierPredictProba",
    "MockClassifierFullTags",
    "MockDeepClusterer",
    "MockSegmenter",
    "SupervisedMockSegmenter",
]

from aeon.testing.mock_estimators._mock_classifiers import (
    MockClassifier,
    MockClassifierFullTags,
    MockClassifierPredictProba,
)
from aeon.testing.mock_estimators._mock_clusterers import MockDeepClusterer
from aeon.testing.mock_estimators._mock_forecasters import (
    MockForecaster,
    MockUnivariateForecasterLogger,
    make_mock_estimator,
)
from aeon.testing.mock_estimators._mock_segmenters import (
    MockSegmenter,
    SupervisedMockSegmenter,
)
