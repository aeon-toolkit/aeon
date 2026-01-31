"""Submodule for similarity search on subsequences."""

__all__ = [
    "BaseSubsequenceSearch",
    "BaseDistanceProfileSearch",
    "MASS",
    "BruteForce",
]

from aeon.similarity_search.subsequence._base import (
    BaseDistanceProfileSearch,
    BaseSubsequenceSearch,
)
from aeon.similarity_search.subsequence._dummy import BruteForce
from aeon.similarity_search.subsequence._mass import MASS
