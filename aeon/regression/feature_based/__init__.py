"""Feature based time series regressors.

While a bit vague, the contents mostly consist of transformers that extract features
pipelined to a vector regressor.
"""

__all__ = [
    "Catch22Regressor",
    "FreshPRINCERegressor",
    "SummaryRegressor",
    "TSFreshRegressor",
]

from aeon.regression.feature_based._catch22 import Catch22Regressor
from aeon.regression.feature_based._fresh_prince import FreshPRINCERegressor
from aeon.regression.feature_based._summary import SummaryRegressor
from aeon.regression.feature_based._tsfresh import TSFreshRegressor
