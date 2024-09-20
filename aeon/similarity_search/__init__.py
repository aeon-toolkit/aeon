"""Similarity search module."""

__all__ = ["BaseSimilaritySearch", "QuerySearch", "SeriesSearch"]

from aeon.similarity_search.base import BaseSimilaritySearch
from aeon.similarity_search.query_search import QuerySearch
from aeon.similarity_search.series_search import SeriesSearch
