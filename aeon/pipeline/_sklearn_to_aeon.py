"""Sklearn to aeon coercion utility."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["sklearn_to_aeon"]

from aeon.pipeline._make_pipeline import make_pipeline
from aeon.transformations.collection import Tabularizer


def sklearn_to_aeon(estimator):
    """Coerces an sklearn estimator to the aeon pipeline interface.

    Creates a pipeline of two elements, the Tabularizer transformer and the estimator.
    The Tabularizer transformer acts as adapter and holds aeon base class logic, as
    well as converting aeon datatypes to a feature vector format. Multivariate series
    will be concatenated into a single feature vector. Data must be of equal length.

    Parameters
    ----------
    estimator : sklearn compatible estimator
        Can be a classifier, regressor, clusterer, or transformer.

    Returns
    -------
    pipe : aeon pipeline estimator
        A pipeline of the Tabularizer transformer and input estimator.
    """
    return make_pipeline(Tabularizer(), estimator)
