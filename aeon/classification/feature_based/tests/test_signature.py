"""Test summary classifier."""

import pytest
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.feature_based import SignatureClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("esig", severity="none"),
    reason="skip test if required soft dependency esig not available",
)
def test_signature_classifier():
    """Test the SignatureClassifier."""
    X, y = make_example_3d_numpy()
    cls = SignatureClassifier(estimator=None)
    cls._fit(X, y)
    assert isinstance(cls.pipeline.named_steps["classifier"], RandomForestClassifier)


@pytest.mark.skipif(
    not _check_soft_dependencies("esig", severity="none"),
    reason="skip test if required soft dependency esig not available",
)
@pytest.mark.parametrize("class_weight", ["balanced", "balanced_subsample"])
def test_signature_classifier_with_class_weight(class_weight):
    """Test signature classifier with class weight."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=12, return_y=True, random_state=0
    )
    clf = SignatureClassifier(
        estimator=RandomForestClassifier(n_estimators=5),
        random_state=0,
        class_weight=class_weight,
    )
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))
