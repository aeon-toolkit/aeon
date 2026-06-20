"""Learned Shapelets tests."""

import numpy as np
import pytest

from aeon.classification.shapelet_based import LearningShapeletClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


def test_learning_shapelet_classifier_is_deprecated():
    """LearningShapeletClassifier raises a deprecation warning."""
    with pytest.warns(
        FutureWarning,
        match="LearningShapeletClassifier is deprecated and will be removed",
    ):
        LearningShapeletClassifier(max_iter=10)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tslearn", "tensorflow"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_get_transform():
    """Learned Shapelets tests not covered by standard test suite."""
    X = make_example_3d_numpy(return_y=False, n_cases=10, n_timepoints=20)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Test get transform and location with and without save_transformed_data
    with pytest.warns(FutureWarning, match="LearningShapeletClassifier"):
        clf = LearningShapeletClassifier(
            max_iter=10, total_lengths=1, save_transformed_data=True
        )
    with pytest.raises(ValueError):
        clf.get_transform(X)
    with pytest.raises(ValueError):
        clf.get_locations(X)
    clf.fit(X, y)
    t = clf.get_transform(X)
    assert isinstance(t, np.ndarray)
