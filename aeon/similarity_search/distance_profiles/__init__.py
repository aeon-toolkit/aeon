# -*- coding: utf-8 -*-
"""Distance profiles."""

__author__ = ["baraline"]
__all__ = ["naive_euclidean_profile", "normalized_naive_euclidean_profile"]

from aeon.similarity_search.distance_profiles.naive_euclidean import (
    naive_euclidean_profile,
)
from aeon.similarity_search.distance_profiles.normalized_naive_euclidean import (
    normalized_naive_euclidean_profile,
)
