# -*- coding: utf-8 -*-
"""Register of estimator base classes corresponding to aeon scitypes.

This module exports the following:

---

BASE_CLASS_REGISTER - list of tuples

each tuple corresponds to a base class, elements as follows:
    0 : string - scitype shorthand
    1 : type - the base class itself
    2 : string - plain English description of the scitype

---

BASE_CLASS_SCITYPE_LIST - list of string
    elements are 0-th entries of BASE_CLASS_REGISTER, in same order

---

BASE_CLASS_LIST - list of string
    elements are 1-st entries of BASE_CLASS_REGISTER, in same order

---

BASE_CLASS_LOOKUP - dictionary
    keys/entries are 0/1-th entries of BASE_CLASS_REGISTER

"""

__author__ = ["fkiraly"]

import pandas as pd

from aeon.annotation.base import BaseSeriesAnnotator
from aeon.base import BaseEstimator, BaseObject
from aeon.classification.base import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.clustering.base import BaseClusterer
from aeon.forecasting.base import BaseForecaster
from aeon.forecasting.model_selection._split import BaseSplitter
from aeon.networks.base import BaseDeepNetwork
from aeon.param_est.base import BaseParamFitter
from aeon.performance_metrics.base import BaseMetric
from aeon.regression.base import BaseRegressor
from aeon.transformations.base import BaseTransformer

BASE_CLASS_REGISTER = [
    ("object", BaseObject, "object"),
    ("estimator", BaseEstimator, "estimator = object with fit"),
    ("classifier", BaseClassifier, "time series classifier"),
    ("clusterer", BaseClusterer, "time series clusterer"),
    ("early_classifier", BaseEarlyClassifier, "early time series classifier"),
    ("forecaster", BaseForecaster, "forecaster"),
    ("metric", BaseMetric, "performance metric"),
    ("network", BaseDeepNetwork, "deep learning network"),
    ("param_est", BaseParamFitter, "parameter fitting estimator"),
    ("regressor", BaseRegressor, "time series regressor"),
    ("series-annotator", BaseSeriesAnnotator, "time series annotator"),
    ("splitter", BaseSplitter, "time series splitter"),
    ("transformer", BaseTransformer, "time series transformer"),
]


BASE_CLASS_SCITYPE_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[0].tolist()

BASE_CLASS_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[1].tolist()

BASE_CLASS_LOOKUP = dict(zip(BASE_CLASS_SCITYPE_LIST, BASE_CLASS_LIST))
