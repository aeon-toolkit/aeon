"""Tests for TimeTransformerClassifier."""

import pytest

from aeon.classification.deep_learning import TimeTransformerClassifier
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_time_transformer_classifier_construct():
    """Test TimeTransformerClassifier construction."""
    model = TimeTransformerClassifier(n_layers=1, n_heads=2, d_model=16, d_inner=32)
    assert model.n_layers == 1
    assert model.n_heads == 2
    assert model.d_model == 16
    assert model.d_inner == 32


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_time_transformer_classifier_fit_predict():
    """Test TimeTransformerClassifier fit and predict."""
    from aeon.datasets import load_unit_test

    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")

    # Use small model for fast test
    model = TimeTransformerClassifier(
        n_layers=1, n_heads=2, d_model=16, d_inner=32, n_epochs=5, batch_size=4
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
