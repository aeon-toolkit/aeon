# -*- coding: utf-8 -*-
"""ShapeDTW test code."""

from sklearn.metrics import accuracy_score

from aeon.classification.distance_based import ShapeDTW
from aeon.datasets import load_unit_test


def test_shape_dtw_compound():
    """Test of ShapeDTW compound."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train BOSS
    dtw = ShapeDTW(
        shape_descriptor_function="compound",
    )
    dtw.fit(X_train, y_train)

    # test train estimate
    preds = dtw.predict(X_train)
    assert accuracy_score(y_train, preds) >= 0.6
