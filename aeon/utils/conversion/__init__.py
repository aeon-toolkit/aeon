"""Conversion utilities."""

__all__ = [
    "get_n_cases",
    "get_type",
    "equal_length",
    "is_equal_length",
    "has_missing",
    "is_univariate",
    "is_nested_univ_dataframe",
    "resolve_equal_length_inner_type",
    "resolve_unequal_length_inner_type",
    "convert_collection",
]

from aeon.utils.conversion._convert_collection import is_nested_univ_dataframe
from aeon.utils.conversion.collection import (
    convert_collection,
    get_n_cases,
    get_type,
    has_missing,
    is_equal_length,
    is_univariate,
    resolve_equal_length_inner_type,
    resolve_unequal_length_inner_type,
)
