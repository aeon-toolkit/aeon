"""Mock query search useful for testing and debugging.

Used in tests for the query search base class.
"""

from aeon.similarity_search.query_search import BaseQuerySearch


class MockQuerySearch(BaseQuerySearch):
    """Mock query search for testing base class predict."""

    def _predict(self, distance_profile, exclusion_size=None):
        """Predict dummy."""
        return [(0, 0)]
