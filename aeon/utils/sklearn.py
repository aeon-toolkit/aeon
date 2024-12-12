"""Sklearn related typing and inheritance checking utility."""

__maintainer__ = []
__all__ = [
    "is_sklearn_estimator",
    "sklearn_estimator_identifier",
    "is_sklearn_transformer",
    "is_sklearn_classifier",
    "is_sklearn_regressor",
    "is_sklearn_clusterer",
]

from inspect import isclass

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from aeon.base import BaseAeonEstimator


def is_sklearn_estimator(obj):
    """Check whether obj is an sklearn estimator.

    Parameters
    ----------
    obj : any class or object

    Returns
    -------
    is_sklearn_est : bool, whether obj is an sklearn estimator (class or instance)
    """
    if not isclass(obj):
        obj = type(obj)

    is_in_sklearn = issubclass(obj, BaseEstimator)
    is_in_aeon = issubclass(obj, BaseAeonEstimator)

    is_sklearn_est = is_in_sklearn and not is_in_aeon
    return is_sklearn_est


mixin_to_identifier = {
    ClassifierMixin: "classifier",
    ClusterMixin: "clusterer",
    RegressorMixin: "regressor",
    TransformerMixin: "transformer",
}


def sklearn_estimator_identifier(obj, var_name="obj"):
    """Return sklearn identifier.

    Parameters
    ----------
    obj : any class or object
    var_name : str, optional, default = "obj"
        name of variable (obj) to display in error message

    Returns
    -------
    str, the sklearn identifier of obj, inferred from inheritance tree, one of
        "classifier" - supervised classifier
        "clusterer" - unsupervised clusterer
        "regressor" - supervised regressor
        "transformer" - transformer (pipeline element, feature extractor, unsupervised)
        "estimator" - sklearn estimator of indeterminate type

    Raises
    ------
    TypeError if obj is not an sklearn estimator, according to is_sklearn_estimator
    """
    if not is_sklearn_estimator(obj):
        raise TypeError(f"{var_name} is not an sklearn estimator, has type {type(obj)}")
    # deal with sklearn pipelines: type is determined by the last element
    if isinstance(obj, Pipeline) or hasattr(obj, "steps"):
        obj = obj.steps[-1][1]
    # deal with generic composites: type is type of wrapped "estimator"
    if isinstance(obj, (GridSearchCV, RandomizedSearchCV)) or hasattr(obj, "estimator"):
        obj = obj.estimator

    # first check whether obj class inherits from sklearn mixins
    sklearn_mixins = tuple(mixin_to_identifier.keys())

    if not isclass(obj):
        obj_class = type(obj)
    else:
        obj_class = obj
    if issubclass(obj_class, sklearn_mixins):
        for mx in sklearn_mixins:
            if issubclass(obj_class, mx):
                return mixin_to_identifier[mx]

    # fallback - estimator of indeterminate type
    return "estimator"


def is_sklearn_transformer(obj):
    """Check whether obj is an sklearn transformer.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn transformer
    """
    return (
        is_sklearn_estimator(obj) and sklearn_estimator_identifier(obj) == "transformer"
    )


def is_sklearn_classifier(obj):
    """Check whether obj is an sklearn classifier.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn classifier
    """
    return (
        is_sklearn_estimator(obj) and sklearn_estimator_identifier(obj) == "classifier"
    )


def is_sklearn_regressor(obj):
    """Check whether obj is an sklearn regressor.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn regressor
    """
    return (
        is_sklearn_estimator(obj) and sklearn_estimator_identifier(obj) == "regressor"
    )


def is_sklearn_clusterer(obj):
    """Check whether obj is an sklearn clusterer.

    Parameters
    ----------
    obj : any object

    Returns
    -------
    bool, whether obj is an sklearn clusterer
    """
    return (
        is_sklearn_estimator(obj) and sklearn_estimator_identifier(obj) == "clusterer"
    )
