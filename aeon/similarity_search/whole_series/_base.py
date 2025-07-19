"""Base class for whole series similarity search."""

__maintainer__ = ["baraline"]
__all__ = [
    "BaseWholeSeriesSearch",
]

from aeon.similarity_search.base import BaseSimilaritySearch


class BaseWholeSeriesSearch(BaseSimilaritySearch):
    """Base class for similarity search applications."""

    def __init__(self):
        super().__init__()
