import numpy as np
import pytest

pytest.importorskip("esig")


from aeon.classification.feature_based import SignatureClassifier


def _make_simple_data():
    X = np.random.rand(10, 5, 1)
    y = np.array([0] * 8 + [1] * 2)
    return X, y


def test_signature_classifier_balanced_class_weight():
    X, y = _make_simple_data()
    clf = SignatureClassifier(class_weight="balanced", random_state=0)
    clf.fit(X, y)


def test_signature_classifier_dict_class_weight():
    X, y = _make_simple_data()
    clf = SignatureClassifier(class_weight={0: 1, 1: 5}, random_state=0)
    clf.fit(X, y)


def test_signature_classifier_invalid_class_weight():
    with pytest.raises(ValueError):
        SignatureClassifier(class_weight="invalid")
