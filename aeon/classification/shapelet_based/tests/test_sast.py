"""SAST tests."""

from sklearn.ensemble import RandomForestClassifier

from aeon.classification.shapelet_based import SASTClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_sast():
    """SAST tests."""
    X, y = make_example_3d_numpy(n_cases=10)

    clf = SASTClassifier(max_iter=50, classifier=RandomForestClassifier())
    clf.fit(X, y)
