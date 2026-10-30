"""Similarity search module."""

__all__ = [
    "BaseSimilaritySearch",
    "BaseSubsequenceSearch",
    "BaseWholeSeriesSearch",
    "subsequence",
    "whole_series",
]

from aeon.similarity_search import subsequence, whole_series
from aeon.similarity_search._base import BaseSimilaritySearch
from aeon.similarity_search.subsequence._base import BaseSubsequenceSearch
from aeon.similarity_search.whole_series._base import BaseWholeSeriesSearch
