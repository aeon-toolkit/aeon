"""RDST tests."""

from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.shapelet_based import RDSTClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def test_predict_proba():
    """RDST tests for code not covered by standard tests."""
    X = make_example_3d_numpy(return_y=False, n_cases=10)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    clf = RDSTClassifier(estimator=RandomForestClassifier(n_jobs=1))
    clf.fit(X, y)
    p = clf._predict_proba(X)
    assert p.shape == (10, 2)
    p = clf._predict(X)
    assert p.shape == (10,)


def test_rdst_estimator_attribute_lifecycle():
    """Test that estimator_ is only set after fitting, not before."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=20)
    model = RDSTClassifier(max_shapelets=5)

    assert not hasattr(model, "estimator_")

    model.fit(X, y)

    assert hasattr(model, "estimator_")
    assert not hasattr(model, "_estimator")


def test_rdst_calls_lapack_check_with_default_estimator():
    """RDSTClassifier should call check_lapack_svd_safe when estimator is None."""
    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    clf = RDSTClassifier(max_shapelets=5)

    with patch(
        "aeon.classification.shapelet_based._rdst.check_lapack_svd_safe"
    ) as mock_check:
        clf.fit(X, y)

    mock_check.assert_called_once()
    args, _ = mock_check.call_args
    assert args[2] == "RDSTClassifier"


def test_rdst_skips_lapack_check_with_custom_estimator():
    """RDSTClassifier should not call check_lapack_svd_safe with a custom estimator."""
    from sklearn.linear_model import RidgeClassifier

    X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12)
    clf = RDSTClassifier(max_shapelets=5, estimator=RidgeClassifier())

    with patch(
        "aeon.classification.shapelet_based._rdst.check_lapack_svd_safe"
    ) as mock_check:
        clf.fit(X, y)

    mock_check.assert_not_called()
