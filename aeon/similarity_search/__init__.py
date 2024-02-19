"""BaseSimilaritySearch."""

__all__ = [
    "BaseSimiliaritySearch",
    "TopKSimilaritySearch",
    "get_speedup_function_names",
]

from aeon.similarity_search.base import (
    BaseSimiliaritySearch,
    get_speedup_function_names,
)
from aeon.similarity_search.top_k_similarity import TopKSimilaritySearch
