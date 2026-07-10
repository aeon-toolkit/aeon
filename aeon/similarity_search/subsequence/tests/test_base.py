"""Tests for subsequence similarity search base classes."""

__maintainer__ = ["baraline"]

import numpy as np
import pytest

from aeon.similarity_search.subsequence._commons import (
    _extract_top_k_from_dist_profile,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_series,
    make_example_3d_numpy,
)
from aeon.testing.mock_estimators._mock_similarity_searchers import (
    MockDistanceProfileSearch,
    MockSubsequenceSearch,
)


class TestBaseSubsequenceSearch:
    """Tests for BaseSubsequenceSearch."""

    # The fit-3D / predict-2D input-shape test lives in
    # ``similarity_search/tests/test_base.py`` (it covers both mock families), so it
    # is not duplicated here.

    def test_check_query_length_valid(self):
        """Test that valid query length passes."""
        length = 10
        estimator = MockSubsequenceSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=2, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=2, n_timepoints=length)
        estimator.fit(X_fit)
        # Should not raise
        indexes, _ = estimator.predict(query)
        assert indexes.shape[0] <= 1

    def test_check_query_length_invalid(self):
        """Test that invalid query length raises ValueError via _check_query_length."""
        length = 10
        estimator = MockSubsequenceSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(
            n_channels=1, n_timepoints=15
        )  # Wrong length
        estimator.fit(X_fit)
        # Directly test the _check_query_length method
        with pytest.raises(ValueError, match="Expected X to have 10 timepoints"):
            estimator._check_query_length(query)

    def test_fit_stores_metadata(self):
        """Test that fit stores correct metadata."""
        estimator = MockSubsequenceSearch(length=10)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=2, n_timepoints=50, return_y=False
        )
        estimator.fit(X_fit)
        assert estimator.n_cases_ == 5
        assert estimator.n_channels_ == 2
        assert estimator.n_timepoints_ == 50

    @pytest.mark.parametrize("length", [0, -3, 55, 2.5, True])
    def test_fit_invalid_length_raises(self, length):
        """Length not an int in [1, n_timepoints_] must raise at fit."""
        estimator = MockSubsequenceSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=3, n_channels=1, n_timepoints=50, return_y=False
        )
        with pytest.raises(ValueError, match="length"):
            estimator.fit(X_fit)

    def test_fit_valid_length_boundaries(self):
        """Length equal to 1 and to n_timepoints_ are both valid."""
        n_timepoints = 30
        X_fit = make_example_3d_numpy(
            n_cases=3,
            n_channels=1,
            n_timepoints=n_timepoints,
            return_y=False,
        )
        MockSubsequenceSearch(length=1).fit(X_fit)
        MockSubsequenceSearch(length=n_timepoints).fit(X_fit)


class TestBaseDistanceProfileSearch:
    """Tests for BaseDistanceProfileSearch."""

    def test_predict_returns_correct_shape(self):
        """Test that predict returns arrays with correct shapes."""
        length = 10
        k = 3
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
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
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
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
        X_fit = make_example_3d_numpy(
            n_cases=10, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        for k in [1, 3, 5]:
            indexes, distances = estimator.predict(query, k=k)
            assert len(indexes) <= k
            assert len(distances) <= k

    @pytest.mark.parametrize("k", [0, -1, 2.5])
    def test_predict_invalid_k_raises(self, k):
        """Invalid k (non-positive / non-int) must raise a clear ValueError."""
        estimator = MockDistanceProfileSearch(length=10)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=10)
        estimator.fit(X_fit)
        with pytest.raises(ValueError, match="k must be a positive integer"):
            estimator.predict(query, k=k)

    def test_predict_k_inf_returns_all_candidates(self):
        """k=np.inf returns every admissible candidate for the subsequence family.

        This closes a family asymmetry: whole-series NaiveSeriesSearch already
        accepted ``np.inf`` while subsequence search crashed.
        """
        length = 10
        n_cases, n_timepoints = 4, 40
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=n_cases,
            n_channels=1,
            n_timepoints=n_timepoints,
            return_y=False,
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        # allow_trivial_matches so every candidate is admissible (no exclusion zone).
        indexes, distances = estimator.predict(
            query, k=np.inf, allow_trivial_matches=True
        )
        n_candidates = n_timepoints - length + 1
        assert len(indexes) == n_cases * n_candidates
        assert len(distances) == len(indexes)

    def test_predict_with_dist_threshold(self):
        """Test that dist_threshold filters matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        # Use a very small threshold
        indexes, distances = estimator.predict(query, k=10, dist_threshold=0.001)
        # All returned distances should be below threshold
        assert np.all(distances <= 0.001)

    def test_predict_exclusion_zone(self):
        """Test that exclusion zones prevent trivial matches."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=1, n_channels=1, n_timepoints=50, return_y=False
        )  # Single series
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
        X_fit = make_example_3d_numpy(
            n_cases=1, n_channels=1, n_timepoints=50, return_y=False
        )
        query = X_fit[0, :, 20 : 20 + length].copy()
        estimator.fit(X_fit)
        # With trivial matches allowed, can get adjacent positions
        indexes, distances = estimator.predict(query, k=10, allow_trivial_matches=True)
        assert len(indexes) >= 1

    def test_predict_X_index_exclusion(self):
        """Test that X_index excludes the query's own location."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
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

    def test_predict_X_index_excludes_last_subsequence(self):
        """Query as the LAST subsequence with X_index must not self-match.

        Regression for the inclusive-vs-exclusive off-by-one in the self-exclusion
        mask: the last candidate position ``n_timepoints - length`` used never to be
        masked, so ``predict`` returned the query's own location at distance ~0.
        """
        length = 10
        n_timepoints = 50
        last = n_timepoints - length  # index of the last candidate subsequence
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=1,
            n_channels=1,
            n_timepoints=n_timepoints,
            return_y=False,
        )
        query = X_fit[0, :, last : last + length].copy()
        estimator.fit(X_fit)
        indexes, distances = estimator.predict(
            query, k=1, X_index=(0, last), exclusion_factor=0.5
        )
        # The trivial self-match (0, last) at distance ~0 must be excluded.
        assert len(indexes) == 1
        assert not (indexes[0, 0] == 0 and indexes[0, 1] == last)
        assert distances[0] > 1e-8

    def test_predict_X_index_invalid_case(self):
        """Test that invalid X_index case raises ValueError."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        with pytest.raises(ValueError, match="out of bounds"):
            estimator.predict(query, k=1, X_index=(10, 0))  # Invalid case index

    @pytest.mark.parametrize("bad_timepoint", [-1, 41, 500])
    def test_predict_X_index_out_of_range_timepoint_raises(self, bad_timepoint):
        """Out-of-range X_index timepoint raises a ValueError.

        Valid candidate positions are ``0 <= i_timepoint <= n_timepoints - length``
        (= 40 here). Previously an out-of-range timepoint silently masked nothing and
        the trivial self-match leaked through.
        """
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        with pytest.raises(ValueError, match="timepoint"):
            estimator.predict(query, k=1, X_index=(0, bad_timepoint))

    @pytest.mark.parametrize("bad_index", [(0, 0, 0), (0,), 5, (0, 1.5), (1.0, 2)])
    def test_predict_X_index_wrong_type_raises(self, bad_index):
        """Non-int / wrong-length X_index raises a TypeError."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        with pytest.raises(TypeError):
            estimator.predict(query, k=1, X_index=bad_index)

    def test_predict_inverse_distance(self):
        """inverse_distance returns the farthest match (global distance argmax)."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)

        # Original (non-inverted) distance profile.
        profile = estimator.compute_distance_profile(query)
        argmax_flat = int(np.argmax(profile))
        exp_case, exp_ts = divmod(argmax_flat, profile.shape[1])

        idx_far, _ = estimator.predict(
            query, k=1, inverse_distance=True, allow_trivial_matches=True
        )
        idx_near, _ = estimator.predict(
            query, k=1, inverse_distance=False, allow_trivial_matches=True
        )

        # The farthest match must be the position that MAXIMIZES the original
        # distance profile, and must differ from the nearest match.
        assert idx_far[0, 0] == exp_case
        assert idx_far[0, 1] == exp_ts
        assert profile[idx_far[0, 0], idx_far[0, 1]] == profile.max()
        assert profile[idx_near[0, 0], idx_near[0, 1]] == profile.min()
        assert (idx_far[0] != idx_near[0]).any()

    def test_compute_distance_profile_shape(self):
        """Test that compute_distance_profile returns correct shape."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=5, n_channels=2, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=2, n_timepoints=length)
        estimator.fit(X_fit)
        dist_profiles = estimator.compute_distance_profile(query)
        expected_candidates = 50 - length + 1
        assert dist_profiles.shape == (5, expected_candidates)

    def test_distances_are_sorted(self):
        """Test that returned distances are in ascending order."""
        length = 10
        estimator = MockDistanceProfileSearch(length=length)
        X_fit = make_example_3d_numpy(
            n_cases=10, n_channels=1, n_timepoints=50, return_y=False
        )
        query = make_example_2d_numpy_series(n_channels=1, n_timepoints=length)
        estimator.fit(X_fit)
        indexes, distances = estimator.predict(query, k=5)
        # Distances should be sorted ascending
        assert np.all(np.diff(distances) >= 0)


def _reference_top_k(dist_profile, k, threshold, allow_trivial_matches, exclusion_size):
    """Independent reference for ``_extract_top_k_from_dist_profile``.

    Walks a single global ascending sort of the flattened profile and applies the
    inclusive per-case exclusion zone on the fly. This is the naive O(n log n)
    specification the optimized extraction must reproduce exactly. On distinct-valued
    profiles the returned index/distance lists are fully determined; on tied profiles
    only the multiset of distances is well defined (unstable argsort).
    """
    n_cases, n_candidates = dist_profile.shape
    flat = dist_profile.ravel()
    order = np.argsort(flat, kind="stable")
    idxs = []
    dists = []
    zones = []  # (case, lb, ub) inclusive
    if not allow_trivial_matches:
        for flat_idx in order:
            i_case, i_ts = divmod(int(flat_idx), n_candidates)
            if any(c == i_case and lb <= i_ts <= ub for (c, lb, ub) in zones):
                continue
            if flat[flat_idx] > threshold:
                break
            idxs.append((i_case, i_ts))
            dists.append(flat[flat_idx])
            zones.append(
                (
                    i_case,
                    max(i_ts - exclusion_size, 0),
                    min(i_ts + exclusion_size, n_candidates - 1),
                )
            )
            if len(idxs) == k:
                break
    else:
        for flat_idx in order:
            if flat[flat_idx] > threshold:
                break
            i_case, i_ts = divmod(int(flat_idx), n_candidates)
            idxs.append((i_case, i_ts))
            dists.append(flat[flat_idx])
            if len(idxs) == k:
                break
    return np.array(idxs, dtype=np.int64).reshape(-1, 2), np.array(dists)


@pytest.mark.parametrize("k", [1, 3, 7, 10**6])
@pytest.mark.parametrize("threshold", [np.inf, 0.5])
@pytest.mark.parametrize("allow_trivial_matches", [False, True])
@pytest.mark.parametrize("exclusion_size", [0, 3, 5])
def test_top_k_extraction_matches_reference(
    k, threshold, allow_trivial_matches, exclusion_size
):
    """The optimized top-k extraction equals the naive reference.

    On distinct-valued (tie-free) profiles both the returned distances and the
    ``(i_case, i_timestamp)`` indexes are fully determined, so we assert exact
    equality of both against the reference implementation.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_cases = int(rng.integers(1, 5))
        n_candidates = int(rng.integers(8, 40))
        dist_profile = rng.random((n_cases, n_candidates))
        # Skip the (astronomically unlikely) tied draw so index equality is defined.
        if len(np.unique(dist_profile)) != dist_profile.size:
            continue
        kk = min(k, n_cases * n_candidates)

        got_idx, got_dist = _extract_top_k_from_dist_profile(
            dist_profile.copy(), kk, threshold, allow_trivial_matches, exclusion_size
        )
        exp_idx, exp_dist = _reference_top_k(
            dist_profile, kk, threshold, allow_trivial_matches, exclusion_size
        )
        np.testing.assert_array_equal(got_dist, exp_dist)
        np.testing.assert_array_equal(got_idx, exp_idx)


def test_top_k_extraction_invariants_on_self_similar():
    """Exclusion-zone semantics hold when top candidates are excluded.

    Constant blocks create many equal minima whose exclusion zones repeatedly reject
    the next-best candidate -- the worst case the optimized extraction targets. Ties
    make the exact returned indexes (and hence which equal-distance values survive)
    ambiguous under an unstable argsort, so instead of comparing to the reference we
    assert the structural invariants the optimized extraction must preserve: each
    returned index holds its reported distance, distances are ascending, and same-case
    matches respect the inclusive exclusion zone.
    """
    exclusion_size = 4
    rng = np.random.default_rng(1)
    for _ in range(20):
        n_cases = int(rng.integers(1, 5))
        n_candidates = int(rng.integers(10, 40))
        dist_profile = rng.random((n_cases, n_candidates))
        # Force a block of identical minima to trigger repeated exclusion passes.
        dist_profile[:, :6] = 0.0
        for k in (1, 5, n_cases * n_candidates):
            got_idx, got_dist = _extract_top_k_from_dist_profile(
                dist_profile.copy(), k, np.inf, False, exclusion_size
            )
            assert np.all(np.diff(got_dist) >= 0)
            # Every returned index must actually hold its reported distance.
            for (i_case, i_ts), d in zip(got_idx, got_dist):
                assert dist_profile[i_case, i_ts] == d
            # Same-case matches are at least exclusion_size + 1 apart (inclusive zone).
            for case in range(n_cases):
                ts = np.sort(got_idx[got_idx[:, 0] == case, 1])
                assert np.all(np.diff(ts) > exclusion_size)
