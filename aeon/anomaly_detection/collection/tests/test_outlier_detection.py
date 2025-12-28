"""Adapter-specific tests for OutlierDetectionAdapter.

Tests focus exclusively on adapter logic: label translation from
sklearn/PyOD conventions (-1 for outlier, 1 for inlier) to aeon
conventions (1 for anomaly, 0 for normal).
"""

__maintainer__ = []

import numpy as np
import pytest
from sklearn.base import BaseEstimator, OutlierMixin

from aeon.anomaly_detection.collection import OutlierDetectionAdapter
from aeon.testing.data_generation import make_example_3d_numpy


class DummyOutlierDetector(BaseEstimator, OutlierMixin):
    """Mock outlier detector for testing adapter label translation."""

    def __init__(self, forced_predictions=None):
        self.forced_predictions = forced_predictions

    def fit(self, X, y=None):
        """Mock fit method."""
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Mock predict returning sklearn convention (-1 outlier, 1 inlier)."""
        if self.forced_predictions is not None:
            return np.array(self.forced_predictions)
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.array([(-1 if i % 2 == 0 else 1) for i in range(n_samples)])


def test_outlier_adapter_label_translation():
    """Test adapter converts sklearn labels to aeon anomaly labels.

    Adapter Logic Under Test:
    -------------------------
    sklearn/PyOD convention:
        -1 = outlier (detected anomaly by the detector)
         1 = inlier (normal sample)

    aeon convention (expected mapping):
        1 = anomaly (should flag as anomaly)
        0 = normal (not an anomaly)

    Translation:
        -1 → 1 (outlier becomes anomaly)
         1 → 0 (inlier becomes normal)
    """
    # Test with mixed pattern
    forced_output = np.array([-1, 1, -1, 1, -1])
    dummy = DummyOutlierDetector(forced_predictions=forced_output)
    adapter = OutlierDetectionAdapter(detector=dummy)

    X = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, return_y=False, random_state=42
    )

    adapter.fit(X)
    predictions = adapter.predict(X)

    # Verify translation: -1 → 1, 1 → 0
    # Input:    [-1,  1, -1,  1, -1]
    # Expected: [ 1,  0,  1,  0,  1]
    expected = np.array([1, 0, 1, 0, 1])
    np.testing.assert_array_equal(predictions, expected)


@pytest.mark.parametrize(
    "input_pattern,expected_output",
    [
        ([-1, -1, -1, -1], [1, 1, 1, 1]),  # All outliers → all anomalies (1)
        ([1, 1, 1, 1], [0, 0, 0, 0]),  # All inliers → all normal (0)
    ],
    ids=["all_outliers", "all_inliers"],
)
def test_outlier_adapter_edge_cases(input_pattern, expected_output):
    """Test adapter handles edge cases correctly.

    Edge cases verify:
    - All outliers (-1) → All anomalies (1)
    - All inliers (1) → All normal (0)
    """
    dummy = DummyOutlierDetector(forced_predictions=input_pattern)
    adapter = OutlierDetectionAdapter(detector=dummy)

    X = make_example_3d_numpy(
        n_cases=len(input_pattern),
        n_channels=1,
        n_timepoints=10,
        return_y=False,
        random_state=42,
    )

    adapter.fit(X)
    predictions = adapter.predict(X)

    np.testing.assert_array_equal(predictions, expected_output)


def test_outlier_adapter_with_real_isolation_forest():
    """Test OutlierDetectionAdapter with real sklearn IsolationForest.

    This test verifies interoperability with actual sklearn estimators.
    """
    from sklearn.ensemble import IsolationForest

    # Create real IsolationForest detector
    detector = IsolationForest(n_estimators=10, random_state=42)
    adapter = OutlierDetectionAdapter(detector=detector)

    # Generate 3D collection data
    X_train = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=10, return_y=False, random_state=42
    )
    X_test = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=10, return_y=False, random_state=43
    )

    # Fit and predict
    adapter.fit(X_train)
    predictions = adapter.predict(X_test)

    # Verify output shape and binary values
    assert predictions.shape == (10,)
    assert np.all(np.isin(predictions, [0, 1]))


def test_outlier_adapter_invalid_estimator_error_on_fit():
    """Test that invalid estimator raises ValueError on fit, not init."""
    # Invalid estimator (string)
    invalid_detector = "not_an_estimator"
    adapter = OutlierDetectionAdapter(detector=invalid_detector)

    # Should NOT error during init - error comes on fit
    X = make_example_3d_numpy(n_cases=10, return_y=False, random_state=42)

    # Now fit should raise ValueError
    with pytest.raises(ValueError, match="must be an outlier detection algorithm"):
        adapter.fit(X)
