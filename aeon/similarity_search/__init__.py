"""Similarity search module."""

__all__ = [
    "DummyQuerySearch",
    "TopKQuerySearch",
]

from aeon.similarity_search.query_search.dummy import DummyQuerySearch
from aeon.similarity_search.query_search.top_k import TopKQuerySearch
