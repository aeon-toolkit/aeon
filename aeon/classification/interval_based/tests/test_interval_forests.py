"""Test interval forest classifiers."""

import re

import numpy as np
import pytest

from aeon.classification.interval_based import (
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    RandomIntervalSpectralEnsembleClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.testing.utils.estimator_checks import _assert_predict_probabilities
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_temporal_importance_curves


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "cls",
    [
        CanonicalIntervalForestClassifier,
        DrCIFClassifier,
        SupervisedTimeSeriesForest,
        TimeSeriesForestClassifier,
    ],
)
def test_tic_curves(cls):
    """Test whether temporal_importance_curves runs without error."""
    import matplotlib

    matplotlib.use("Agg")

    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"base_estimator": ContinuousIntervalTree()})

    clf = cls(**params)
    clf.fit(X_train, y_train)

    names, curves = clf.temporal_importance_curves()
    plot_temporal_importance_curves(curves, names)


@pytest.mark.parametrize("cls", [RandomIntervalSpectralEnsembleClassifier])
def test_tic_curves_invalid(cls):
    """Test whether temporal_importance_curves raises an error."""
    clf = cls()
    with pytest.raises(
        NotImplementedError, match="No temporal importance curves available."
    ):
        clf.temporal_importance_curves()


@pytest.mark.skipif(
    not _check_soft_dependencies(["pycatch22"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("cls", [CanonicalIntervalForestClassifier, DrCIFClassifier])
def test_forest_pycatch22(cls):
    """Test whether the forest classifiers with pycatch22 run without error."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]
    X_test, _ = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["test"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"use_pycatch22": True})

    clf = cls(**params)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)
    _assert_predict_probabilities(prob, X_test, n_classes=2)


@pytest.mark.parametrize(
    ("verbose", "expected_output", "excluded_output"),
    [
        (1, "[DrCIF] Progress: built=", "[DrCIF] Estimator 1/"),
        (2, "[DrCIF] Estimator 1/", "[DrCIF] Progress: built="),
    ],
)
def test_drcif_fit_verbosity_levels(verbose, expected_output, excluded_output, capsys):
    """DrCIF verbosity controls periodic or per-estimator fit output."""
    n_cases = 20
    n_timepoints = 24
    n_estimators = 2
    X, y = make_example_3d_numpy(
        n_cases=n_cases, n_timepoints=n_timepoints, n_labels=2, random_state=0
    )
    drcif = DrCIFClassifier(
        n_estimators=n_estimators,
        n_intervals=2,
        att_subsample_size=2,
        random_state=0,
        verbose=verbose,
    )

    drcif.fit(X, y)
    output = capsys.readouterr().out

    assert f"[DrCIF] Starting fit: n_cases={n_cases}" in output
    assert expected_output in output
    assert excluded_output not in output
    assert f"[DrCIF] Finished fit: built={n_estimators}" in output
    if verbose == 2:
        assert "estimated_remaining=" in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "remaining_time_pattern"),
    [
        (1, r"contract_remaining=\d+\.\d+s"),
        (2, r"contract_remaining=\d+m \d+s"),
        (120, r"contract_remaining=\d+h \d+m"),
    ],
)
def test_drcif_contract_verbosity_reports_remaining_time(
    time_limit_in_minutes, remaining_time_pattern, capsys
):
    """DrCIF level-two output reports the remaining fit contract."""
    n_cases = 20
    n_timepoints = 24
    max_n_estimators = 2
    X, y = make_example_3d_numpy(
        n_cases=n_cases, n_timepoints=n_timepoints, n_labels=2, random_state=0
    )
    drcif = DrCIFClassifier(
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_estimators=max_n_estimators,
        n_intervals=2,
        att_subsample_size=2,
        random_state=0,
        verbose=2,
    )

    drcif.fit(X, y)
    output = capsys.readouterr().out

    assert re.search(remaining_time_pattern, output)
    assert "estimated_remaining=" not in output
    assert f"[DrCIF] Finished fit: built={max_n_estimators}" in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "expect_progress"),
    [(1e-12, True), (120, False)],
)
def test_drcif_contract_level_one_progress_is_rate_limited(
    time_limit_in_minutes, expect_progress, capsys
):
    """DrCIF level-one contract progress is emitted only after its interval."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_timepoints=24, n_labels=2, random_state=0
    )
    drcif = DrCIFClassifier(
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        random_state=0,
        verbose=1,
    )

    drcif.fit(X, y)
    output = capsys.readouterr().out

    assert ("[DrCIF] Progress: built=" in output) is expect_progress
    assert "[DrCIF] Estimator " not in output


def test_drcif_parallel_verbosity_preserves_fit(capsys):
    """Parallel DrCIF fits are unchanged when detailed output is enabled."""
    n_cases = 20
    n_timepoints = 24
    n_estimators = 4
    n_jobs = 2
    X, y = make_example_3d_numpy(
        n_cases=n_cases, n_timepoints=n_timepoints, n_labels=2, random_state=0
    )
    params = {
        "n_estimators": n_estimators,
        "n_intervals": 2,
        "att_subsample_size": 2,
        "n_jobs": n_jobs,
        "parallel_backend": "threading",
        "random_state": 0,
    }

    quiet = DrCIFClassifier(**params).fit(X, y)
    detailed = DrCIFClassifier(**params, verbose=2).fit(X, y)
    output = capsys.readouterr().out

    assert detailed._n_jobs == n_jobs
    assert f"[DrCIF] Estimator 1/{n_estimators}:" in output
    np.testing.assert_array_equal(quiet.predict_proba(X), detailed.predict_proba(X))
