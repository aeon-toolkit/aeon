"""Register of estimator base classes corresponding to aeon class.

This module exports the following:

---

BASE_CLASS_REGISTER - list of tuples

each tuple corresponds to a base class, elements as follows:
    0 : string - shorthand identifier for base class type
    1 : type - the base class itself
    2 : string - plain English description of the class

---

BASE_CLASS_IDENTIFIER_LIST - list of string
    elements are 0-th entries of BASE_CLASS_REGISTER, in same order

---

BASE_CLASS_LIST - list of string
    elements are 1-st entries of BASE_CLASS_REGISTER, in same order

---

BASE_CLASS_LOOKUP - dictionary
    keys/entries are 0/1-th entries of BASE_CLASS_REGISTER

"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "BASE_CLASS_REGISTER",
    "VALID_ESTIMATOR_BASES",
]


from aeon.anomaly_detection.base import BaseAnomalyDetector
from aeon.base import BaseAeonEstimator, BaseCollectionEstimator, BaseSeriesEstimator
from aeon.classification.base import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering.base import BaseClusterer
from aeon.regression.base import BaseRegressor
from aeon.segmentation.base import BaseSegmenter
from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer

# all base classes
BASE_CLASS_REGISTER = {
    # abstract - no estimator directly inherits from these
    "collection-estimator": BaseCollectionEstimator,
    "estimator": BaseAeonEstimator,
    "series-estimator": BaseSeriesEstimator,
    "transformer": BaseTransformer,
    # estimator types
    "anomaly-detector": BaseAnomalyDetector,
    "collection-transformer": BaseCollectionTransformer,
    "classifier": BaseClassifier,
    "clusterer": BaseClusterer,
    "early_classifier": BaseEarlyClassifier,
    "regressor": BaseRegressor,
    "segmenter": BaseSegmenter,
    "similarity_searcher": BaseSimilaritySearch,
    "series-transformer": BaseSeriesTransformer,
}

# base classes which are valid for estimator to directly inherit from
VALID_ESTIMATOR_BASES = {
    k: BASE_CLASS_REGISTER[k]
    for k in BASE_CLASS_REGISTER.keys()
    - {"estimator", "collection-estimator", "series-estimator", "transformer"}
}
