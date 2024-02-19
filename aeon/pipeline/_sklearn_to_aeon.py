"""Sklearn to aeon coercion utility."""

__maintainer__ = []

from aeon.pipeline._make_pipeline import make_pipeline


def sklearn_to_aeon(estimator):
    """Coerces an sklearn estimator to the aeon pipeline interface.

    Creates a pipeline of two elements, the identity transformer and the estimator.
    The identity transformer acts as adapter and holds aeon base class logic.

    Developer note:
    Type dispatch logic is in the transformer base class, in the `__mul__` dunder.

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
    from aeon.transformations.compose import Id

    return make_pipeline(Id(), estimator)
