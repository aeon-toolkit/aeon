"""Sklearn to aeon coercion utility."""

__maintainer__ = ["MatthewMiddlehurst"]

from aeon.pipeline._make_pipeline import make_pipeline
from aeon.transformations.collection import Tabularizer


def sklearn_to_aeon(estimator):
    """Coerces an sklearn estimator to the aeon pipeline interface.

    Creates a pipeline of two elements, the identity transformer and the estimator.
    The identity transformer acts as adapter and holds aeon base class logic.

    Parameters
    ----------
    estimator : sklearn compatible estimator
        can be classifier, regressor, transformer, clusterer

    Returns
    -------
    pipe : aeon estimator of corresponding time series type
        classifiers, regressors, clusterers are converted to time series counterparts
        by flattening time series. Assumes equal length time series.
        transformers are converted to time series transformer by application per series
    """
    return make_pipeline(Tabularizer(), estimator)
