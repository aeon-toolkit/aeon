"""Rotation Forest test code."""

import re

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.sklearn import RotationForestClassifier
from aeon.datasets import load_unit_test


def test_rotf_output():
    """Test of RotF contracting and train estimate on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=10,
        pca_solver="randomized",
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    expected = [
        [0.8, 0.2],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.9, 0.1],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.4, 0.6],
    ]

    np.testing.assert_array_almost_equal(
        expected, rotf.predict_proba(X_test[:15]), decimal=4
    )


def test_contracted_rotf():
    """Test of RotF contracting and train estimate on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2d")

    rotf = RotationForestClassifier(
        time_limit_in_minutes=5,
        contract_max_n_estimators=5,
        random_state=0,
    )
    rotf.fit(X_train, y_train)
    assert len(rotf.estimators_) > 0

    y_pred = rotf.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y_test)

    acc = accuracy_score(y_test, y_pred)
    np.testing.assert_almost_equal(acc, 0.909, decimal=4)


def test_rotf_fit_predict():
    """Test of RotF fit_predict on test data."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")

    rotf = RotationForestClassifier(
        n_estimators=5,
        random_state=0,
    )

    y_proba = rotf.fit_predict_proba(X_train, y_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), 2)
    assert len(rotf.estimators_) > 0
    assert rotf._is_fitted

    y_proba = rotf.predict_proba(X_train)
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (len(y_train), 2)


def test_rotf_input():
    """Test RotF with incorrect input."""
    rotf = RotationForestClassifier()

    X = rotf._check_X(np.random.random((10, 1, 100)))
    assert X.shape == (10, 100)

    with pytest.raises(
        ValueError, match="RotationForestClassifier is not a time series classifier"
    ):
        rotf._check_X(np.random.random((10, 10, 100)))
    with pytest.raises(
        ValueError, match="RotationForestClassifier is not a time series classifier"
    ):
        rotf._check_X([[1, 2, 3], [4, 5], [6, 7, 8]])

    X2 = np.zeros((10, 10))
    y = np.zeros(10)
    y[0:5] = 1

    with pytest.raises(ValueError, match="All attributes in X contain the same value."):
        rotf.fit_predict(X2, y)


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_rotf_verbosity_levels(verbose, capsys):
    """RotationForest verbosity controls summary and per-estimator progress."""
    X, y = load_unit_test(split="train", return_type="numpy2d")
    rotf = RotationForestClassifier(
        n_estimators=2,
        random_state=0,
        verbose=verbose,
    )

    rotf.fit(X, y)
    output = capsys.readouterr().out

    if verbose == 0:
        assert output == ""
    else:
        assert "[RotF] Starting fit:" in output
        assert "[RotF] Finished fit: built=2" in output
        if verbose == 1:
            assert "[RotF] Progress: built=" in output
            assert "[RotF] Estimator " not in output
        else:
            assert "[RotF] Estimator 1/2:" in output
            assert "estimated_remaining=" in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "remaining_time_pattern"),
    [
        (1, r"contract_remaining=\d+(?:\.\d+)?s"),
        (2, r"contract_remaining=\d+m \d+s"),
        (120, r"contract_remaining=\d+h \d+m"),
    ],
)
def test_rotf_contract_verbosity_reports_remaining_time(
    time_limit_in_minutes, remaining_time_pattern, capsys
):
    """RotationForest level-two output formats the remaining fit contract."""
    X, y = load_unit_test(split="train", return_type="numpy2d")
    rotf = RotationForestClassifier(
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_estimators=1,
        random_state=0,
        verbose=2,
    )

    rotf.fit(X, y)
    output = capsys.readouterr().out

    assert "[RotF] Estimator 1:" in output
    assert re.search(remaining_time_pattern, output)
    assert "estimated_remaining=" not in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "reports_progress"),
    [(0.0005, True), (120, False)],
)
def test_rotf_contract_level_one_progress_is_rate_limited(
    time_limit_in_minutes, reports_progress, capsys
):
    """RotationForest level one limits output for short completed contracts."""
    X, y = load_unit_test(split="train", return_type="numpy2d")
    rotf = RotationForestClassifier(
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_estimators=1,
        random_state=0,
        verbose=1,
    )

    rotf.fit(X, y)
    output = capsys.readouterr().out

    assert ("[RotF] Progress: built=" in output) is reports_progress
    assert "[RotF] Estimator " not in output


def test_parallel_verbose_fit_preserves_predictions(capsys):
    """Verbose batching preserves deterministic parallel RotationForest output."""
    X, y = load_unit_test(split="train", return_type="numpy2d")
    quiet = RotationForestClassifier(
        n_estimators=4,
        n_jobs=2,
        random_state=0,
    ).fit(X, y)
    verbose = RotationForestClassifier(
        n_estimators=4,
        n_jobs=2,
        random_state=0,
        verbose=2,
    ).fit(X, y)

    np.testing.assert_allclose(verbose.predict_proba(X), quiet.predict_proba(X))
    assert verbose._n_jobs == 2
    output = capsys.readouterr().out
    assert "[RotF] Estimator 1/4:" in output
    assert "[RotF] Estimator 4/4:" in output
