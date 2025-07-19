"""Base class for subsequence similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseSubsequenceSearch",
]

from aeon.similarity_search.base import BaseSimilaritySearch


class BaseSubsequenceSearch(BaseSimilaritySearch):
    """Base class for similarity search applications."""

    def __init__(self, length: int):
        self.length = length
        super().__init__()
