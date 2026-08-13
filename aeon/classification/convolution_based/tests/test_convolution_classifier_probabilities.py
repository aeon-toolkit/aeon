"""Tests for convolutional classifier probability fallbacks."""

import numpy as np
import pytest

from aeon.classification.convolution_based import (
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


def _assert_one_hot_predictions(classifier, X):
    probabilities = classifier.predict_proba(X[:5])
    predictions = classifier.predict(X[:5])
    expected = np.zeros_like(probabilities)
    expected[
        np.arange(len(predictions)),
        np.searchsorted(classifier.classes_, predictions),
    ] = 1
    np.testing.assert_array_equal(probabilities, expected)


@pytest.mark.parametrize(
    ("classifier_class", "n_kernels"),
    [(MiniRocketClassifier, 5), (MultiRocketClassifier, 100)],
)
def test_rocket_classifier_probability_fallback(classifier_class, n_kernels):
    """MiniRocket-family classifiers return one-hot probabilities."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=20, random_state=0
    )
    classifier = classifier_class(n_kernels=n_kernels, random_state=0).fit(X, y)

    _assert_one_hot_predictions(classifier, X)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch is not available",
)
@pytest.mark.parametrize(
    "classifier_class", [HydraClassifier, MultiRocketHydraClassifier]
)
def test_hydra_classifier_probability_fallback(classifier_class):
    """Hydra classifiers return one-hot probabilities from their predictions."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=20, random_state=0
    )
    classifier = classifier_class(n_kernels=5, n_groups=4, random_state=0).fit(X, y)

    _assert_one_hot_predictions(classifier, X)
