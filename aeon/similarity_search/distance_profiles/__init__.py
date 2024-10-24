"""Distance profiles."""

__all__ = [
    "euclidean_distance_profile",
    "normalized_euclidean_distance_profile",
    "squared_distance_profile",
    "normalized_squared_distance_profile",
]


from aeon.similarity_search.distance_profiles.euclidean_distance_profile import (
    euclidean_distance_profile,
    normalized_euclidean_distance_profile,
)
from aeon.similarity_search.distance_profiles.squared_distance_profile import (
    normalized_squared_distance_profile,
    squared_distance_profile,
)
