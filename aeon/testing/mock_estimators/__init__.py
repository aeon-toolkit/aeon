"""Mock forecasters for testing and debugging."""

__all__ = [
    "MockForecaster",
    "MockUnivariateForecasterLogger",
    "make_mock_estimator",
    "MockClassifier",
    "MockClassifierPredictProba",
    "MockClassifierFullTags",
    "MockClassifierMultiTestParams",
    "MockDeepClusterer",
    "MockSegmenter",
    "SupervisedMockSegmenter",
    "MockHandlesAllInput",
    "MockRegressor",
    "MockMultivariateSeriesTransformer",
    "MockSeriesTransformerNoFit",
    "MockUnivariateSeriesTransformer",
    "MockTransformer",
    "MockCollectionTransformer",
    "MockSeriesTransformer",
]

from aeon.testing.mock_estimators._mock_classifiers import (
    MockClassifier,
    MockClassifierFullTags,
    MockClassifierMultiTestParams,
    MockClassifierPredictProba,
)
from aeon.testing.mock_estimators._mock_clusterers import MockDeepClusterer
from aeon.testing.mock_estimators._mock_collection_transformers import (
    MockCollectionTransformer,
)
from aeon.testing.mock_estimators._mock_forecasters import (
    MockForecaster,
    MockUnivariateForecasterLogger,
    make_mock_estimator,
)
from aeon.testing.mock_estimators._mock_regressors import (
    MockHandlesAllInput,
    MockRegressor,
)
from aeon.testing.mock_estimators._mock_segmenters import (
    MockSegmenter,
    SupervisedMockSegmenter,
)
from aeon.testing.mock_estimators._mock_series_transformers import (
    MockMultivariateSeriesTransformer,
    MockSeriesTransformer,
    MockSeriesTransformerNoFit,
    MockUnivariateSeriesTransformer,
)
from aeon.testing.mock_estimators._mock_transformers import MockTransformer
