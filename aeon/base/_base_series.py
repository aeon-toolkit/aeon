"""Base class for estimators that fit single (possibly multivariate) time series."""

from aeon.base._base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps


class BaseSeriesEstimator(BaseEstimator):
    """Base class for estimators that use single (possibly multivariate) time series.

    Provides functions that are common to BaseSegmenter,
    BaseSeriesTransformer for the checking and
    conversion of input to fit, predict and transform, where relevant.

    It also stores the common default tags used by all the subclasses and meta data
    describing the characteristics of time series passed to ``fit``.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "X_inner_type": "numpy3D",
        "capability:multithreading": False,
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }

    def __init__(self):
        super(BaseSeriesEstimator, self).__init__()
        _check_estimator_deps(self)
