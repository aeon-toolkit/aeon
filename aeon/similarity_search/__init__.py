"""BaseSimilaritySearch."""

__all__ = [
    "BaseSeriesSimilaritySearch",
    "BaseCollectionSimiliaritySearch",
    "TopKSimilaritySearch",
    "get_speedup_function_names",
]

from aeon.similarity_search.base import (
    BaseCollectionSimiliaritySearch,
    BaseSeriesSimilaritySearch,
    get_speedup_function_names,
)
from aeon.similarity_search.top_k_similarity import TopKSimilaritySearch
