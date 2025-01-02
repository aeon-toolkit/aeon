"""Subsequence search module."""

__all__ = [
    "BaseSubsequenceSearch",
    "BaseMatrixProfile",
    "StompMatrixProfile",
    "BruteForceMatrixProfile",
]

from aeon.similarity_search.subsequence_search._brute_force import (
    BruteForceMatrixProfile,
)
from aeon.similarity_search.subsequence_search._stomp import StompMatrixProfile
from aeon.similarity_search.subsequence_search.base import (
    BaseMatrixProfile,
    BaseSubsequenceSearch,
)
