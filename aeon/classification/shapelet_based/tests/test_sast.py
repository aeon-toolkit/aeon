"""SAST tests."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.shapelet_based import RSASTClassifier, SASTClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


def test_predict_proba():
    """SAST tests for code not covered by standard tests."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    clf = SASTClassifier(classifier=RandomForestClassifier())
    clf.fit(X, y)
    p = clf._predict_proba(X)
    assert p.shape == (10, 2)
    try:
        clf = RSASTClassifier(classifier=RandomForestClassifier())
        clf.fit(X, y)
        p = clf._predict_proba(X)
        assert p.shape == (10, 2)
    except ModuleNotFoundError:
        pass


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_most_important_feature_on_ts():
    """Test whether test_plot_most_important_feature_on_ts runs without error."""
    import matplotlib.pyplot as plt

    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    clf = SASTClassifier()
    clf.fit(X, y)
    fig = clf.plot_most_important_feature_on_ts(X[0][0], y)
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()
