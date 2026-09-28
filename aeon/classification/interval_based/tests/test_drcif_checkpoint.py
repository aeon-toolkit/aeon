"""DrCIF continuation preserves interval selectors, trees and OOB estimates."""

from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash

from aeon.base._estimators.interval_based.base_interval_forest import _run_jobs
from aeon.classification.interval_based import DrCIFClassifier
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.fixture
def training_data():
    """Small multivariate data exercising all three DrCIF representations."""
    return make_example_3d_numpy(
        n_cases=12, n_channels=2, n_timepoints=24, random_state=42
    )


def _assert_same_forest(expected, actual, X):
    assert actual.n_estimators_ == actual._n_estimators == expected.n_estimators_
    assert len(actual.estimators_) == len(actual.intervals_) == actual.n_estimators_
    np.testing.assert_array_equal(expected.predict_proba(X), actual.predict_proba(X))
    for expected_tree, actual_tree in zip(expected.intervals_, actual.intervals_):
        for expected_rep, actual_rep in zip(expected_tree, actual_tree):
            assert joblib_hash(expected_rep.intervals_) == joblib_hash(
                actual_rep.intervals_
            )
    assert joblib_hash(expected._rng) == joblib_hash(actual._rng)
    assert joblib_hash(expected._train_rng) == joblib_hash(actual._train_rng)
    assert joblib_hash(expected._train_estimates) == joblib_hash(
        actual._train_estimates
    )


@pytest.mark.parametrize("contracted, train_estimates", [(False, False), (True, True)])
def test_drcif_resume(training_data, checkpoint_directory, contracted, train_estimates):
    """Loaded forests grow and truncate with aligned interval and OOB state."""
    X, y = training_data
    initial_size, target_size = 1, 3
    seed = np.random.RandomState(0)
    seed_state = joblib_hash(seed)
    params = dict(n_intervals=2, att_subsample_size=2, random_state=seed)
    limit = "contract_max_n_estimators" if contracted else "n_estimators"
    if contracted:
        params["time_limit_in_minutes"] = 5
    full = DrCIFClassifier(**params, **{limit: target_size})
    partial = DrCIFClassifier(**params, **{limit: initial_size})
    method = "fit_predict_proba" if train_estimates else "fit"
    expected = getattr(full, method)(X, y)
    getattr(partial, method)(X, y)
    initial_predictions = partial.predict_proba(X)
    assert joblib_hash(seed) == seed_state
    path = checkpoint_directory / "drcif.pkl"
    partial.save_checkpoint(path)
    restored = DrCIFClassifier.load_checkpoint(path)
    restored.set_params(**{limit: target_size}, n_jobs=2, parallel_backend="threading")
    assert restored.resume_fit(X, y) is restored
    _assert_same_forest(full, restored, X)
    if train_estimates:
        np.testing.assert_array_equal(expected, restored._checkpoint_train_proba())
    restored.set_params(**{limit: initial_size}).resume_fit(X, y)
    assert restored.n_estimators_ == initial_size
    assert len(restored.intervals_) == initial_size
    assert len(restored._train_estimates) == (initial_size if train_estimates else 0)
    np.testing.assert_array_equal(initial_predictions, restored.predict_proba(X))


@pytest.mark.parametrize("train_estimates", [False, True])
def test_drcif_interrupted_batch(training_data, checkpoint_directory, train_estimates):
    """An interrupted parallel batch resumes without skipping tree or OOB seeds."""
    X, y = training_data
    n_jobs = 2
    target_size = n_jobs + 1
    params = dict(
        n_estimators=target_size,
        n_intervals=2,
        att_subsample_size=2,
        random_state=0,
        n_jobs=n_jobs,
        parallel_backend="threading",
    )
    method = "fit_predict_proba" if train_estimates else "fit"
    full = DrCIFClassifier(**params)
    getattr(full, method)(X, y)
    path = checkpoint_directory / "drcif.pkl"
    interval_minutes = 1
    partial = DrCIFClassifier(
        **params, checkpoint_path=path, checkpoint_interval=interval_minutes
    )
    calls = 0

    def fail_next_batch(tasks, *args, **kwargs):
        nonlocal calls
        calls += 1
        # Training estimates add a second job submission to each batch.
        if calls == (3 if train_estimates else 2):
            raise RuntimeError("interrupted")
        return _run_jobs(tasks, *args, **kwargs)

    with (
        patch(
            "aeon.base._estimators.interval_based.base_interval_forest._run_jobs",
            fail_next_batch,
        ),
        patch("aeon.base._checkpoint.time") as clock,
    ):
        clock.monotonic.side_effect = count(step=interval_minutes * 60)
        with pytest.raises(RuntimeError, match="interrupted"):
            getattr(partial, method)(X, y)
    restored = DrCIFClassifier.load_checkpoint(path)
    assert not restored.is_fitted
    assert restored.n_estimators_ == n_jobs
    restored.resume_fit(X, y)
    _assert_same_forest(full, restored, X)
    assert DrCIFClassifier.load_checkpoint(path).is_fitted


def test_drcif_contract_budget(training_data):
    """Resume spends the remaining budget; fit resets it and honours the tree cap."""
    X, y = training_data
    initial_size, target_size = 1, 3
    budget_minutes = 5
    classifier = DrCIFClassifier(
        n_intervals=2,
        att_subsample_size=2,
        time_limit_in_minutes=budget_minutes,
        contract_max_n_estimators=initial_size,
        n_jobs=2,
        parallel_backend="threading",
        random_state=0,
    ).fit(X, y)
    assert classifier.n_estimators_ == initial_size
    classifier.set_params(contract_max_n_estimators=target_size)
    classifier.fit_elapsed_time_ = budget_minutes * 60
    classifier.resume_fit(X, y)
    assert classifier.n_estimators_ == initial_size
    classifier.set_params(time_limit_in_minutes=budget_minutes * 2).resume_fit(X, y)
    assert classifier.n_estimators_ == target_size
    classifier.fit_elapsed_time_ = budget_minutes * 2 * 60
    classifier.fit(X, y)
    assert classifier.n_estimators_ == target_size
