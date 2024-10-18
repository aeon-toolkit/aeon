"""Validation functions for tags."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "check_valid_tags",
    "check_tag_value",
]

from inspect import isclass

from aeon.base import BaseAeonEstimator
from aeon.utils.base import BASE_CLASS_REGISTER
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
    if isinstance(estimator, BaseAeonEstimator):
        if tags is None:
            tags = estimator.get_tags()
        method = isinstance
        est_name = estimator.__class__.__name__
    elif isclass(estimator) and issubclass(estimator, BaseAeonEstimator):
        if tags is None:
            tags = estimator.get_class_tags()
        method = issubclass
        est_name = estimator.__name__
    else:
        raise ValueError(
            "Estimator must be an instance or subclass of BaseAeonEstimator."
        )

    for tag_name, tag_value in tags.items():
        # check if the tag exists
        if tag_name not in ESTIMATOR_TAGS:
            raise ValueError(
                f"Tag {tag_name} is not a valid tag, it does not exist in the "
                f"aeon.utils.tags.ESTIMATOR_TAGS dictionary."
            )
        tag = ESTIMATOR_TAGS[tag_name]

        # check if the tag is compatible with the estimator class
        tag_classes = tag["class"] if isinstance(tag["class"], list) else [tag["class"]]
        compatible_class = False
        for tag_class in tag_classes:
            if method(estimator, BASE_CLASS_REGISTER[tag_class]):
                compatible_class = True
                break

        if not compatible_class:
            raise ValueError(
                f"Tag {tag_name} is not compatible with the estimator class "
                f"{est_name}. It is only compatible with the following "
                f"classes: {tag_classes}."
            )

        # check if the tag value is valid
        check_tag_value(tag_name, tag_value, raise_error=True)

    if error_on_missing:
        tag_names = all_tags_for_estimator(estimator, names_only=True)
        missing_tags = set(tag_names) - set(tags.keys())
        if missing_tags:
            raise ValueError(
                f"Tags {missing_tags} are missing from the estimator {est_name}."
            )


def check_tag_value(tag, value, raise_error=False):
    """Check if a value is valid for a tag.

    Parameters
    ----------
    tag : str
        The tag to check the value for.
    value : object
        The value to check.
    raise_error : bool
        If True, raise an error if the value is invalid.

    Returns
    -------
    bool
        True if the value is valid, False otherwise.
    """
    if tag not in ESTIMATOR_TAGS:
        raise ValueError(
            f"Tag {tag} is not a valid tag, it does not exist in the "
            f"aeon.utils.tags.ESTIMATOR_TAGS dictionary."
        )
    tag_info = ESTIMATOR_TAGS[tag]

    tag_types = (
        tag_info["type"] if isinstance(tag_info["type"], list) else [tag_info["type"]]
    )
    valid_value = False
    for tag_type in tag_types:
        if isinstance(tag_type, tuple):
            if (
                isinstance(value, list)
                and tag_type[0] in ["list", "list||str"]
                and all(x in tag_type[1] for x in value)
            ):
                valid_value = True
                break
            elif (
                isinstance(value, str)
                and tag_type[0] in ["str", "list||str"]
                and value in tag_type[1]
            ):
                valid_value = True
                break
        elif tag_type == "list":
            if isinstance(value, list):
                valid_value = True
                break
        elif tag_type == "bool":
            if isinstance(value, bool):
                valid_value = True
                break
        elif tag_type == "str":
            if isinstance(value, str):
                valid_value = True
                break
        elif tag_type is None:
            if value is None:
                valid_value = True
                break

    if raise_error and not valid_value:
        raise ValueError(
            f"Value {value} is not a valid value for tag {tag}. "
            f"Valid values are of type(s) {tag_types}."
        )

    return valid_value
