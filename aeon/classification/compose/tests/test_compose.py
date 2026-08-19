"""Tests for classification compose estimators."""

import numpy as np

from aeon.classification import DummyClassifier
from aeon.classification.compose import ClassifierChannelEnsemble


def test_classifier_channel_ensemble_with_remainder():
    """Test ClassifierChannelEnsemble fits remaining channels with remainder."""
    X = np.random.RandomState(0).rand(10, 3, 5)
    y = np.array([0, 1] * 5)

    clf = ClassifierChannelEnsemble(
        classifiers=[("first", DummyClassifier())],
        channels=[0],
        remainder=DummyClassifier(),
    )

    clf.fit(X, y)

    assert len(clf.ensemble_) == 2
    assert clf.ensemble_[1][0] == "Remainder"
    assert clf.channels_[1] == [1, 2]

    preds = clf.predict(X)
    assert preds.shape == (10,)
