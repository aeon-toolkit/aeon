"""Lookup functions for tags."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["all_tags_for_estimator"]

from aeon.base import BaseEstimator
from aeon.registry import (
    BASE_CLASS_IDENTIFIER_LIST,
    BASE_CLASS_LIST,
    BASE_CLASS_REGISTER,
)
from aeon.utils.tags import ESTIMATOR_TAGS


def all_tags_for_estimator(
    estimator,
    names_only=False,
):
    """Get a filtered dictionary of tags given an estimator or estimator type.

    Retrieves tags directly from ``aeon.utils.tags.ESTIMATOR_TAGS``.

    Parameters
    ----------
    estimator: string, estimator object, estimator class
        Filter tags returned by the type of estimator.
        Valid input strings can be found in the tags dictionary in
        ``aeon.utils.tags.ESTIMATOR_TAGS`` under the class key or in
        ``aeon.registry.BASE_CLASS_REGISTER``.
    names_only: bool, default=False
        If True, return only the names of the tags as a list.

    Returns
    -------
    tags: dictionary of tags,
        filtered version of ``aeon.utils.tags.ESTIMATOR_TAGS``.
    """
    if isinstance(estimator, BaseEstimator):
        method = isinstance
    elif issubclass(estimator, BaseEstimator):
        method = issubclass
    elif isinstance(estimator, str) and estimator in BASE_CLASS_REGISTER:
        estimator = BASE_CLASS_LIST[BASE_CLASS_IDENTIFIER_LIST.index(estimator)]
        method = issubclass
    else:
        raise ValueError(
            "Estimator must be an instance or subclass of BaseEstimator, "
            "or a valid string from BASE_CLASS_REGISTER."
        )

    filtered_tags = {}
    for tag_name, tag in ESTIMATOR_TAGS.items():
        tag_classes = tag["class"] if isinstance(tag["class"], list) else [tag["class"]]
        for tag_class in tag_classes:
            if method(
                estimator, BASE_CLASS_LIST[BASE_CLASS_IDENTIFIER_LIST.index(tag_class)]
            ):
                filtered_tags[tag_name] = tag

    return list(filtered_tags.keys()) if names_only else filtered_tags
