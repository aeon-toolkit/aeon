"""Estimator tags and tag utilities."""

__all__ = [
    "ESTIMATOR_TAGS",
    "check_valid_tags",
    "all_tags_for_estimator",
]

from aeon.utils.tags._discovery import all_tags_for_estimator
from aeon.utils.tags._tags import ESTIMATOR_TAGS
from aeon.utils.tags._validate import check_valid_tags
