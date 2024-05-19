"""Pipeline making utility."""

__maintainer__ = ["MatthewMiddlehurst"]

from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin

from aeon.classification import BaseClassifier
from aeon.classification.compose import ClassifierPipeline
from aeon.clustering import BaseClusterer
from aeon.clustering.compose import ClustererPipeline
from aeon.regression import BaseRegressor
from aeon.regression.compose import RegressorPipeline
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.compose import CollectionTransformerPipeline


def make_pipeline(*steps):
    """Create a pipeline from aeon and sklearn estimators.

    Currently available for:
        forecasters, classifiers, regressors, clusterers, and transformers.

    Parameters
    ----------
    steps : list or tuple of aeon and/or sklearn estimators
        This should be provided in same order as the required pipeline construction

    Returns
    -------
    pipe : aeon pipeline containing steps, in order
        always a descendant of BaseObject, precise object determined by
        equivalent to result of step[0] * step[1] * ... * step[-1]

    Examples
    --------
    Example 1: forecaster pipeline
    >>> from aeon.datasets import load_airline
    >>> from aeon.forecasting.trend import PolynomialTrendForecaster
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.exponent import ExponentTransformer
    >>> pipe = make_pipeline(ExponentTransformer(), PolynomialTrendForecaster())
    >>> type(pipe).__name__
    'TransformedTargetForecaster'

    Example 2: classifier pipeline
    >>> from aeon.classification.feature_based import Catch22Classifier
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.exponent import ExponentTransformer
    >>> pipe = make_pipeline(ExponentTransformer(), Catch22Classifier())
    >>> type(pipe).__name__
    'ClassifierPipeline'

    Example 3: transformer pipeline
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.exponent import ExponentTransformer
    >>> pipe = make_pipeline(ExponentTransformer(), ExponentTransformer())
    >>> type(pipe).__name__
    'TransformerPipeline'
    """
    if len(steps) == 1 and isinstance(steps[0], list):
        steps = steps[0]

    # Classifiers
    if (
        isinstance(steps[-1], BaseClassifier)
        or isinstance(steps[-1], ClassifierMixin)
        or getattr(steps[-1], "_estimator_type", None) == "classifier"
    ):
        return ClassifierPipeline(list(steps[:-1]), steps[-1])
    # Regressors
    elif (
        isinstance(steps[-1], BaseRegressor)
        or isinstance(steps[-1], RegressorMixin)
        or getattr(steps[-1], "_estimator_type", None) == "regressor"
    ):
        return RegressorPipeline(list(steps[:-1]), steps[-1])
    # Clusterers
    elif (
        isinstance(steps[-1], BaseClusterer)
        or isinstance(steps[-1], ClusterMixin)
        or getattr(steps[-1], "_estimator_type", None) == "clusterer"
    ):
        return ClustererPipeline(list(steps[:-1]), steps[-1])
    # Collection transformers
    elif (
        isinstance(steps[0], BaseCollectionTransformer)
        or isinstance(steps[0], TransformerMixin)
        or getattr(steps[0], "_estimator_type", None) == "transformer"
    ) and (
        isinstance(steps[-1], BaseTransformer)
        or isinstance(steps[-1], TransformerMixin)
        or getattr(steps[-1], "_estimator_type", None) == "transformer"
    ):
        return CollectionTransformerPipeline(list(steps))
    else:
        pipe = steps[0]
        for i in range(1, len(steps)):
            pipe = pipe * steps[i]

        return pipe
