"""Implements registry for aeon estimator base classes and tags."""

from aeon.registry._base_classes import (
    BASE_CLASS_IDENTIFIER_LIST,
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    BASE_CLASS_REGISTER,
)
from aeon.registry._identifier import get_identifiers
from aeon.registry._lookup import all_estimators

__all__ = [
    "all_estimators",
    "get_identifiers",
    "BASE_CLASS_REGISTER",
    "BASE_CLASS_LIST",
    "BASE_CLASS_LOOKUP",
    "BASE_CLASS_IDENTIFIER_LIST",
]
