"""Pipeline making utility."""

__maintainer__ = []
from aeon.classification import BaseClassifier
from aeon.classification.compose import ClassifierPipeline


def make_pipeline(*steps):
    """Create a pipeline from estimators of any type.

    Parameters
    ----------
    steps : tuple of aeon estimators
        in same order as used for pipeline construction

    Returns
    -------
    pipe : aeon pipeline containing steps, in order
        always a descendant of BaseObject, precise object determined by
        equivalent to result of step[0] * step[1] * ... * step[-1]

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()

    Example 1: forecaster pipeline
    >>> from aeon.datasets import load_airline
    >>> from aeon.forecasting.trend import PolynomialTrendForecaster
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.exponent import ExponentTransformer
    >>> y = load_airline()
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
    if isinstance(steps[-1], BaseClassifier):
        pipe = ClassifierPipeline(list(steps[:-1]), steps[-1])
    else:
        pipe = steps[0]
        for i in range(1, len(steps)):
            pipe = pipe * steps[i]

    return pipe
