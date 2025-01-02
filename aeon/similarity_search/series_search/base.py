"""Base class for whole series search."""

__maintainer__ = ["baraline"]

from aeon.similarity_search._base import BaseSimilaritySearch


class BaseSeriesSearch(BaseSimilaritySearch):
    """."""

    ...


class BaseIndexSearch(BaseSimilaritySearch):
    """."""

    ...

    def batch_fit(sourcefiles, batch_size):
        """."""
        # fit
        # and then update
