"""Adapter-specific tests for ClassificationAdapter.

Tests focus exclusively on adapter logic: verifying classifier predictions
pass through unchanged (no label translation needed).
"""

__maintainer__ = ["SebastianSchmidl"]

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from aeon.anomaly_detection.collection import ClassificationAdapter
from aeon.testing.data_generation import make_example_3d_numpy


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Mock classifier for testing adapter pass-through behavior."""

    def __init__(self, forced_predictions=None):
        self.forced_predictions = forced_predictions

    def fit(self, X, y):
        """Mock fit method."""
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Mock predict method returning binary class labels (0 or 1)."""
        if self.forced_predictions is not None:
            return np.array(self.forced_predictions)
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.array([i % 2 for i in range(n_samples)])


def test_classification_adapter_passthrough_behavior():
    """Test adapter preserves classifier output without modification."""
    # Test with specific pattern that would reveal any modification
    forced_predictions = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
    dummy_clf = DummyClassifier(forced_predictions=forced_predictions)
    adapter = ClassificationAdapter(classifier=dummy_clf)

    X_train, y_train = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=10, n_labels=2, random_state=42
    )
    X_test = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=10, return_y=False, random_state=43
    )

    adapter.fit(X_train, y_train)
    predictions = adapter.predict(X_test)

    # Verify exact passthrough
    np.testing.assert_array_equal(predictions, forced_predictions)

    # Verify dtype preservation
    assert np.issubdtype(
        predictions.dtype, np.integer
    ), f"Expected int, got {predictions.dtype}"

    # Verify no inversion (counts should match)
    assert np.sum(predictions == 0) == 3, "Label inversion detected"
    assert np.sum(predictions == 1) == 3, "Label inversion detected"


@pytest.mark.parametrize(
    "input_pattern,description",
    [
        ([0, 0, 0, 0], "all_normal"),
        ([1, 1, 1, 1], "all_anomaly"),
    ],
    ids=["all_normal", "all_anomaly"],
)
def test_classification_adapter_edge_cases(input_pattern, description):
    """Test adapter preserves edge case predictions unchanged.

    Edge cases: all normal (0), all anomaly (1).
    """
    forced_predictions = np.array(input_pattern)
    dummy_clf = DummyClassifier(forced_predictions=forced_predictions)
    adapter = ClassificationAdapter(classifier=dummy_clf)

    X_train, y_train = make_example_3d_numpy(n_cases=10, n_labels=2, random_state=42)
    X_test = make_example_3d_numpy(
        n_cases=len(input_pattern), return_y=False, random_state=43
    )

    adapter.fit(X_train, y_train)
    predictions = adapter.predict(X_test)

    np.testing.assert_array_equal(predictions, forced_predictions)
