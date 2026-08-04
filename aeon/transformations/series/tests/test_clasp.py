"""Test ClaSP series transformer."""

import numpy as np

from aeon.datasets import load_gun_point_segmentation
from aeon.transformations.series import ClaSPTransformer
from aeon.transformations.series._clasp import clasp


def test_clasp():
    """Test ClaSP series transformer returned size."""
    for dtype in [np.float64, np.float32, np.float16]:
        series = np.arange(100, dtype=dtype)
        clasp = ClaSPTransformer()
        profile = clasp.fit_transform(series)

        m = len(series) - clasp.window_length + 1
        assert np.float64 == profile.dtype
        assert m == len(profile)


def test_clasp_f1_handles_undefined_class_scores():
    """F1 scoring remains finite when a split has no positives for a class."""
    n_timepoints = 50
    series, window_length, _ = load_gun_point_segmentation()
    series = series[:n_timepoints]
    transformer = ClaSPTransformer(
        window_length=window_length,
        scoring_metric="F1",
    )

    profile = transformer.fit_transform(series)

    expected_length = n_timepoints - window_length + 1
    assert profile.shape == (expected_length,)
    assert np.isfinite(profile).all()


def test_clasp_repeats_available_neighbours_when_k_exceeds_candidates():
    """ClaSP repeats valid neighbours when fewer than requested are available."""
    n_timepoints = 10
    window_length = 9
    k_neighbours = 5
    n_subsequences = n_timepoints - window_length + 1
    series = np.arange(n_timepoints, dtype=np.float64)

    _, knn_mask = clasp(
        series,
        m=window_length,
        k_neighbours=k_neighbours,
        interpolate=False,
    )
    expected_column = np.resize(np.arange(n_subsequences), k_neighbours)
    expected = np.repeat(expected_column[:, np.newaxis], n_subsequences, axis=1)

    np.testing.assert_array_equal(knn_mask, expected)
