"""Tests for class_weight support in SignatureClassifier."""

import numpy as np
import pytest

from aeon.classification.feature_based import SignatureClassifier

pytest.importorskip("esig")


def _make_simple_data():
    """Create a simple imbalanced dataset for testing."""
    X = np.random.rand(10, 5, 1)
    y = np.array([0] * 8 + [1] * 2)
    return X, y


def test_signature_classifier_balanced_class_weight():
    """Test SignatureClassifier with balanced class_weight."""
    X, y = _make_simple_data()
    clf = SignatureClassifier(class_weight="balanced", random_state=0)
    clf.fit(X, y)


def test_signature_classifier_dict_class_weight():
    """Test SignatureClassifier with dict-based class_weight."""
    X, y = _make_simple_data()
    clf = SignatureClassifier(class_weight={0: 1, 1: 5}, random_state=0)
    clf.fit(X, y)


def test_signature_classifier_invalid_class_weight():
    """Test SignatureClassifier raises ValueError for invalid class_weight."""
    with pytest.raises(ValueError):
        SignatureClassifier(class_weight="invalid")
