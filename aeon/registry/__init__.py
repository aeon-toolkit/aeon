"""Implements registry for aeon estimator base classes and tags."""

from aeon.registry._base_classes import (
    BASE_CLASS_IDENTIFIER_LIST,
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    BASE_CLASS_REGISTER,
)
from aeon.registry._data_types import (
    ALL_TIME_SERIES_TYPES,
    COLLECTIONS_DATA_TYPES,
    HIERARCHICAL_DATA_TYPES,
    SERIES_DATA_TYPES,
)
from aeon.registry._identifier import get_identifiers
from aeon.registry._lookup import all_estimators, all_tags
from aeon.registry._tags import (
    ESTIMATOR_TAG_LIST,
    ESTIMATOR_TAG_REGISTER,
    check_tag_is_valid,
)

__all__ = [
    "all_estimators",
    "all_tags",
    "check_tag_is_valid",
    "get_identifiers",
    "ESTIMATOR_TAG_LIST",
    "ESTIMATOR_TAG_REGISTER",
    "BASE_CLASS_REGISTER",
    "BASE_CLASS_LIST",
    "BASE_CLASS_LOOKUP",
    "BASE_CLASS_IDENTIFIER_LIST",
    "ALL_TIME_SERIES_TYPES",
    "COLLECTIONS_DATA_TYPES",
    "SERIES_DATA_TYPES",
    "HIERARCHICAL_DATA_TYPES",
]
