"""Common utility functions for whole series similarity search."""

__maintainer__ = ["baraline"]

# Note: The _extract_top_k_from_dist_profile function has been moved to
# aeon.similarity_search.subsequence._commons to avoid duplication.
# It now handles 2D distance profiles (n_cases, n_candidates).
# For whole series search, reshape the 1D profile to (n_cases, 1).
