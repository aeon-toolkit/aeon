"""Tests for RocketClassifier."""

import numpy as np
from sklearn.linear_model import RidgeClassifier

from aeon.classification.convolution_based import RocketClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_rocket_classifier_predict_proba_without_estimator_probabilities():
    """Fallback probabilities should be one-hot and aligned with classes_."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=20, random_state=0
    )
    classifier = RocketClassifier(
        n_kernels=5,
        estimator=RidgeClassifier(),
        random_state=0,
    ).fit(X, y)

    probabilities = classifier.predict_proba(X[:5])
    predictions = classifier.predict(X[:5])
    expected = np.zeros_like(probabilities)
    expected[
        np.arange(len(predictions)), np.searchsorted(classifier.classes_, predictions)
    ] = 1

    np.testing.assert_array_equal(probabilities, expected)
