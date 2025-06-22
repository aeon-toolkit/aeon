"""Register of estimator base classes.

BASE_CLASS_REGISTER - dictionary
    Dictionary of base classes for each estimator type. Keys are identifier strings,
    values are base classes.

VALID_ESTIMATOR_BASES - dictionary
    Dictionary of base classes that are valid for estimators to inherit from. Subset of
    BASE_CLASS_REGISTER. Keys are identifier strings, values are base classes.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "BASE_CLASS_REGISTER",
    "VALID_ESTIMATOR_BASES",
]

from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.anomaly_detection.collection.base import BaseCollectionAnomalyDetector
from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.base import BaseAeonEstimator, BaseCollectionEstimator, BaseSeriesEstimator
from aeon.classification.base import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering.base import BaseClusterer
from aeon.forecasting.base import BaseForecaster
from aeon.regression.base import BaseRegressor
from aeon.segmentation.base import BaseSegmenter
from aeon.similarity_search._base import BaseSimilaritySearch
from aeon.similarity_search.collection import BaseCollectionSimilaritySearch
from aeon.similarity_search.series import BaseSeriesSimilaritySearch
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer

# all base classes
BASE_CLASS_REGISTER = {
    # abstract - no estimator directly inherits from these
    "estimator": BaseAeonEstimator,
    "collection-estimator": BaseCollectionEstimator,
    "series-estimator": BaseSeriesEstimator,
    "transformer": BaseTransformer,
    "anomaly-detector": BaseAnomalyDetector,
    "similarity-search": BaseSimilaritySearch,
    # estimator types
    "collection-anomaly-detector": BaseCollectionAnomalyDetector,
    "collection-similarity-search": BaseCollectionSimilaritySearch,
    "collection-transformer": BaseCollectionTransformer,
    "classifier": BaseClassifier,
    "clusterer": BaseClusterer,
    "early_classifier": BaseEarlyClassifier,
    "forecaster": BaseForecaster,
    "regressor": BaseRegressor,
    "segmenter": BaseSegmenter,
    "series-anomaly-detector": BaseSeriesAnomalyDetector,
    "series-similarity-search": BaseSeriesSimilaritySearch,
    "series-transformer": BaseSeriesTransformer,
}

# base classes which are valid for estimator to directly inherit from
VALID_ESTIMATOR_BASES = {
    k: BASE_CLASS_REGISTER[k]
    for k in BASE_CLASS_REGISTER.keys()
    - {
        "estimator",
        "collection-estimator",
        "series-estimator",
        "transformer",
        "anomaly-detector",
        "similarity-search",
    }
}
