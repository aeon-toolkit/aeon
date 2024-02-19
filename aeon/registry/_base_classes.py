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

__maintainer__ = []

import pandas as pd

from aeon.annotation.base import BaseSeriesAnnotator
from aeon.base import (
    BaseCollectionEstimator,
    BaseEstimator,
    BaseObject,
    BaseSeriesEstimator,
)
from aeon.classification.base import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering.base import BaseClusterer
from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.model_selection._split import BaseSplitter
from aeon.networks.base import BaseDeepNetwork
from aeon.performance_metrics.base import BaseMetric
from aeon.regression.base import BaseRegressor
from aeon.segmentation.base import BaseSegmenter
from aeon.similarity_search.base import BaseSimiliaritySearch
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer

BASE_CLASS_REGISTER = [
    ("object", BaseObject, "object"),
    ("estimator", BaseEstimator, "estimator = object with fit"),
    ("classifier", BaseClassifier, "classifier"),
    ("collection-estimator", BaseCollectionEstimator, "collection estimator"),
    ("collection-transformer", BaseCollectionTransformer, "collection transformer"),
    ("clusterer", BaseClusterer, "clusterer"),
    ("early_classifier", BaseEarlyClassifier, "early time series classifier"),
    ("forecaster", BaseForecaster, "forecaster"),
    ("metric", BaseMetric, "performance metric"),
    ("network", BaseDeepNetwork, "deep learning network"),
    ("regressor", BaseRegressor, "regressor"),
    ("segmenter", BaseSegmenter, "segmenter"),
    ("series-annotator", BaseSeriesAnnotator, "annotator"),
    ("series-estimator", BaseSeriesEstimator, "single series estimator"),
    ("series-transformer", BaseSeriesTransformer, "single series transformer"),
    ("splitter", BaseSplitter, "splitter"),
    ("similarity-search", BaseSimiliaritySearch, "similarity search"),
    ("transformer", BaseTransformer, "transformer"),
]


BASE_CLASS_IDENTIFIER_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[0].tolist()

BASE_CLASS_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[1].tolist()

BASE_CLASS_LOOKUP = dict(zip(BASE_CLASS_IDENTIFIER_LIST, BASE_CLASS_LIST))
