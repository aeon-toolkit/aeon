"""Test ClaSP series transformer."""

import numpy as np
import pytest

from aeon.transformations.series import ClaSPTransformer
from aeon.transformations.series._clasp import (
    _binary_f1_score,
    _compute_distances_iterative,
    _roc_auc_score,
    _sliding_dot_product,
    clasp,
)


def test_clasp():
    """Test ClaSP series transformer returned size."""
    for dtype in [np.float64, np.float32, np.float16]:
        series = np.arange(100, dtype=dtype)
        clasp = ClaSPTransformer()
        profile = clasp.fit_transform(series)

        m = len(series) - clasp.window_length + 1
        assert np.float64 == profile.dtype
        assert m == len(profile)


def test_clasp_f1_scoring_metric():
    """Test ``scoring_metric='F1'`` produces a profile of valid ClaSP length."""
    series = np.arange(100, dtype=np.float64)
    transformer = ClaSPTransformer(scoring_metric="F1")
    profile = transformer.fit_transform(series)

    m = len(series) - transformer.window_length + 1
    assert len(profile) == m


def test_clasp_invalid_scoring_metric_raises():
    """Test fitting rejects scoring metrics outside the supported set."""
    series = np.arange(100, dtype=np.float64)
    transformer = ClaSPTransformer(scoring_metric="bogus")
    with pytest.raises(ValueError, match="invalid input"):
        transformer.fit_transform(series)


def test_clasp_warns_when_window_too_large():
    """Test oversized windows warn before ClaSP profile computation."""
    series = np.arange(50, dtype=np.float64)
    transformer = ClaSPTransformer(window_length=46)
    with pytest.warns(UserWarning, match="larger than size of the time series"):
        transformer.fit_transform(series)


def test_clasp_without_interpolation():
    """Test ``clasp(..., interpolate=False)`` returns aligned profile and kNN mask."""
    profile, knn_mask = clasp(np.arange(50, dtype=np.float64), m=5, interpolate=False)
    assert profile.shape[0] == knn_mask.shape[1]


def test_sliding_dot_product_handles_odd_lengths():
    """Test FFT sliding dot product handles odd and even length parity mixes."""
    query_odd = np.array([1.0, 2.0, 3.0])
    ts_even = np.arange(10, dtype=np.float64)
    result = _sliding_dot_product(query_odd, ts_even)
    assert result.shape[0] == len(ts_even) - len(query_odd) + 1

    query_even = np.array([1.0, 2.0])
    ts_odd = np.arange(11, dtype=np.float64)
    result2 = _sliding_dot_product(query_even, ts_odd)
    assert result2.shape[0] == len(ts_odd) - len(query_even) + 1


def test_compute_distances_iterative_pads_when_fewer_than_k_candidates():
    """Test kNN padding uses valid subsequence indices when fewer than k exist."""
    X = np.arange(10, dtype=np.float64)
    knns = _compute_distances_iterative(X, m=9, k=3, n_jobs=1)
    assert knns.shape == (2, 3)
    assert (knns < 2).all()


def test_binary_f1_score_handles_degenerate_split():
    """Test binary F1 returns a finite score with no predicted positives."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 0, 0, 0], dtype=np.int64)
    score = _binary_f1_score(y_true, y_pred)
    assert 0.0 <= score <= 1.0


def test_roc_auc_score_degenerate_single_class_is_nan():
    """Test ROC-AUC returns nan for a single-class split."""
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    y_true = np.zeros(4, dtype=np.int64)
    assert np.isnan(_roc_auc_score(y_score, y_true))
