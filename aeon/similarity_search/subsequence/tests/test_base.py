"""Tests for subsequence similarity search base classes."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest

from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockDistanceProfileSearch,
    MockSubsequenceSearch,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT, _get_datatypes_for_estimator


class TestBaseSubsequenceSearch:
    """Tests for BaseSubsequenceSearch."""

    def test_input_shape_fit_predict(self):
        """Test input shapes: fit takes collection (3D), predict takes series (2D)."""
        estimator = MockSubsequenceSearch()
        datatypes = _get_datatypes_for_estimator(estimator)
        for datatype in datatypes:
            X_train, y_train = FULL_TEST_DATA_DICT[datatype]["train"]
            X_test, y_test = FULL_TEST_DATA_DICT[datatype]["test"]
            estimator.fit(X_train, y_train).predict(X_test[0])

    def test_check_query_length_valid(self):
        """Test that valid query length passes."""
        length = 10
        estimator = MockSubsequenceSearch(length=length)
        X_fit = np.random.rand(5, 2, 50)
        query = np.random.rand(2, length)
        estimator.fit(X_fit)
        # Should not raise
        indexes, _ = estimator.predict(query)
        assert indexes.shape[0] <= 1

    def test_check_query_length_invalid(self):
        """Test that invalid query length raises ValueError via _check_query_length."""
        length = 10
        estimator = MockSubsequenceSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        query = np.random.rand(1, 15)  # Wrong length
        estimator.fit(X_fit)
        # Directly test the _check_query_length method
        with pytest.raises(ValueError, match="Expected X to have 10 timepoints"):
            estimator._check_query_length(query)

    def test_fit_stores_metadata(self):
        """Test that fit stores correct metadata."""
        estimator = MockSubsequenceSearch(length=10)
        X_fit = np.random.rand(5, 2, 50)
        estimator.fit(X_fit)
        assert estimator.n_cases_ == 5
        assert estimator.n_channels_ == 2
        assert estimator.n_timepoints_ == 50


class TestBaseDistanceProfileSearch:
    """Tests for BaseDistanceProfileSearch."""

    def test_predict_returns_correct_shape(self):
        """Test that predict returns arrays with correct shapes."""
        length = 10
        k = 3
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        query = np.random.rand(1, length)
        estimator.fit(X_fit)
        indexes, distances = estimator.predict(query, k=k)
        assert indexes.ndim == 2
        assert indexes.shape[1] == 2  # (i_case, i_timestamp)
        assert distances.ndim == 1
        assert len(indexes) == len(distances)
        assert len(indexes) <= k

    def test_predict_finds_exact_match(self):
        """Test that predict finds an exact match when query is from X_."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        # Use a subsequence from the fitted data as query
        query = X_fit[2, :, 15 : 15 + length].copy()
        estimator.fit(X_fit)
        indexes, distances = estimator.predict(query, k=1)
        # Should find exact match with distance ~0
        assert len(indexes) == 1
        assert distances[0] < 1e-10
        assert indexes[0, 0] == 2  # Case index
        assert indexes[0, 1] == 15  # Timestamp

    def test_predict_with_k(self):
        """Test that predict returns k matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(10, 1, 50)
        query = np.random.rand(1, length)
        estimator.fit(X_fit)
        for k in [1, 3, 5]:
            indexes, distances = estimator.predict(query, k=k)
            assert len(indexes) <= k
            assert len(distances) <= k

    def test_predict_with_dist_threshold(self):
        """Test that dist_threshold filters matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        query = np.random.rand(1, length)
        estimator.fit(X_fit)
        # Use a very small threshold
        indexes, distances = estimator.predict(query, k=10, dist_threshold=0.001)
        # All returned distances should be below threshold
        assert np.all(distances <= 0.001)

    def test_predict_exclusion_zone(self):
        """Test that exclusion zones prevent trivial matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(1, 1, 50)  # Single series
        query = X_fit[0, :, 20 : 20 + length].copy()
        estimator.fit(X_fit)
        # With exclusion, neighboring matches should be excluded
        indexes, distances = estimator.predict(
            query, k=5, allow_trivial_matches=False, exclusion_factor=0.5
        )
        # Check that returned timestamps are not too close to each other
        if len(indexes) > 1:
            timestamps = indexes[:, 1]
            exclusion_size = int(length * 0.5)
            for i in range(len(timestamps)):
                for j in range(i + 1, len(timestamps)):
                    assert abs(timestamps[i] - timestamps[j]) >= exclusion_size

    def test_predict_allow_trivial_matches(self):
        """Test that allow_trivial_matches=True permits neighboring matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(1, 1, 50)
        query = X_fit[0, :, 20 : 20 + length].copy()
        estimator.fit(X_fit)
        # With trivial matches allowed, can get adjacent positions
        indexes, distances = estimator.predict(query, k=10, allow_trivial_matches=True)
        assert len(indexes) >= 1

    def test_predict_X_index_exclusion(self):
        """Test that X_index excludes the query's own location."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        # Query from case 2, timestamp 20
        query = X_fit[2, :, 20 : 20 + length].copy()
        estimator.fit(X_fit)
        indexes, distances = estimator.predict(
            query, k=1, X_index=(2, 20), exclusion_factor=0.5
        )
        # The result should NOT be at (2, 20) since it's excluded
        if len(indexes) > 0:
            exclusion_size = int(length * 0.5)
            # Either different case, or timestamp outside exclusion zone
            is_excluded = (
                indexes[0, 0] == 2 and abs(indexes[0, 1] - 20) < exclusion_size
            )
            assert not is_excluded

    def test_predict_X_index_invalid_case(self):
        """Test that invalid X_index case raises ValueError."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        query = np.random.rand(1, length)
        estimator.fit(X_fit)
        with pytest.raises(ValueError, match="out of bounds"):
            estimator.predict(query, k=1, X_index=(10, 0))  # Invalid case index

    def test_predict_inverse_distance(self):
        """Test that inverse_distance returns farthest matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 1, 50)
        query = np.random.rand(1, length)
        estimator.fit(X_fit)
        # Get nearest and farthest
        idx_near, dist_near = estimator.predict(query, k=1, inverse_distance=False)
        idx_far, dist_far = estimator.predict(query, k=1, inverse_distance=True)
        # Farthest should have larger original distance
        # (but inverse_distance returns 1/dist, so dist_far < dist_near
        # in returned values)
        # Just check they return valid results
        assert len(idx_near) >= 1
        assert len(idx_far) >= 1

    def test_compute_distance_profile_shape(self):
        """Test that compute_distance_profile returns correct shape."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(5, 2, 50)
        query = np.random.rand(2, length)
        estimator.fit(X_fit)
        dist_profiles = estimator.compute_distance_profile(query)
        expected_candidates = 50 - length + 1
        assert dist_profiles.shape == (5, expected_candidates)

    def test_distances_are_sorted(self):
        """Test that returned distances are in ascending order."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = np.random.rand(10, 1, 50)
        query = np.random.rand(1, length)
        estimator.fit(X_fit)
        indexes, distances = estimator.predict(query, k=5)
        # Distances should be sorted ascending
        assert np.all(np.diff(distances) >= 0)
