"""Validation functions for tags."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["check_valid_tags"]

from aeon.base import BaseEstimator
from aeon.registry import BASE_CLASS_IDENTIFIER_LIST, BASE_CLASS_LIST
from aeon.utils.tags import ESTIMATOR_TAGS, all_tags_for_estimator


def check_valid_tags(estimator, tags=None, error_on_missing=True):
    """Check the validity of tags for an estimator.

    Parameters
    ----------
    estimator : estimator object or class
        The estimator to check tags for.
    tags : dict
        The tags to check. If None, all tags are checked.
    error_on_missing : bool
        If True, raise an error if a tag is missing.

    Raises
    ------
    ValueError
        If a tag is not valid for the estimator.
    """
    if isinstance(estimator, BaseEstimator):
        if tags is None:
            tags = estimator.get_tags()
        method = isinstance
        est_name = estimator.__class__.__name__
    elif issubclass(estimator, BaseEstimator):
        if tags is None:
            tags = estimator.get_class_tags()
        method = issubclass
        est_name = estimator.__name__
    else:
        raise ValueError("Estimator must be an instance or subclass of BaseEstimator.")

    for tag_name, tag_value in tags.items():
        # check if the tag exists
        if tag_name not in ESTIMATOR_TAGS:
            raise ValueError(
                f"Tag {tag_name} is not a valid tag, it does not exist in the "
                f"ESTIMATOR_TAGS dictionary."
            )
        tag = ESTIMATOR_TAGS[tag_name]

        # check if the tag is compatible with the estimator class
        tag_classes = tag["class"] if isinstance(tag["class"], list) else [tag["class"]]
        compatible_class = False
        for tag_class in tag_classes:
            if method(
                estimator, BASE_CLASS_LIST[BASE_CLASS_IDENTIFIER_LIST.index(tag_class)]
            ):
                compatible_class = True
                break

        if not compatible_class:
            raise ValueError(
                f"Tag {tag_name} is not compatible with the estimator class "
                f"{est_name}. It is only compatible with the following "
                f"classes: {tag_classes}."
            )

        # check if the tag value is valid
        tag_types = tag["type"] if isinstance(tag["type"], list) else [tag["type"]]
        compatible_value = False
        for tag_type in tag_types:
            if isinstance(tag_type, tuple):
                if (
                    isinstance(tag_value, list)
                    and tag_type[0] in ["list", "list||str"]
                    and all(x in tag_type[1] for x in tag_value)
                ):
                    compatible_value = True
                    break
                elif (
                    isinstance(tag_value, str)
                    and tag_type[0] in ["str", "list||str"]
                    and tag_value in tag_type[1]
                ):
                    compatible_value = True
                    break
            elif tag_type == "list":
                if isinstance(tag_value, list):
                    compatible_value = True
                    break
            elif tag_type == "bool":
                if isinstance(tag_value, bool):
                    compatible_value = True
                    break
            elif tag_type == "str":
                if isinstance(tag_value, str):
                    compatible_value = True
                    break
            elif tag_type is None:
                if tag_value is None:
                    compatible_value = True
                    break

        if not compatible_value:
            raise ValueError(
                f"Tag {tag_name} has an invalid value. The value {tag_value} is not "
                f"compatible with the tag type(s) {tag['type']}."
            )

    if error_on_missing:
        tag_names = all_tags_for_estimator(estimator, names_only=True)
        missing_tags = set(tag_names) - set(tags.keys())
        if missing_tags:
            raise ValueError(
                f"Tags {missing_tags} are missing from the estimator " f"{est_name}."
            )
