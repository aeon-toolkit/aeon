"""Arsenal continuation tests, including parallel and interrupted training."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash

from aeon.classification.convolution_based import Arsenal
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.estimator_checking import check_estimator
from aeon.testing.mock_estimators import MockClassifier


@pytest.fixture
def checkpoint_directory():
    """Create and clean up an isolated directory for checkpoint files."""
    with TemporaryDirectory() as directory:
        yield Path(directory)


@pytest.fixture
def _training_data():
    """Provide a small deterministic classification dataset."""
    return make_example_3d_numpy(n_cases=12, n_timepoints=24, random_state=42)


def _assert_same_ensemble(expected, actual, X):
    assert actual.n_estimators_ == expected.n_estimators_
    assert len(actual.estimators_) == len(actual.weights_) == actual.n_estimators_
    np.testing.assert_array_equal(actual.weights_, expected.weights_)
    np.testing.assert_array_equal(actual.predict_proba(X), expected.predict_proba(X))
    seeds = [member[0].random_state for member in actual.estimators_]
    assert len(set(seeds)) == len(seeds)
    for expected_member, actual_member in zip(expected.estimators_, actual.estimators_):
        assert joblib_hash(expected_member[0].kernels) == joblib_hash(
            actual_member[0].kernels
        )
    assert joblib_hash(expected._rng.get_state()) == joblib_hash(
        actual._rng.get_state()
    )
    assert joblib_hash(expected._train_estimates) == joblib_hash(
        actual._train_estimates
    )


@pytest.mark.parametrize("contracted, train_estimates", [(False, False), (True, True)])
def test_arsenal_resume(
    _training_data, checkpoint_directory, contracted, train_estimates
):
    """A resumed ensemble has the same members, RNG and OOB state as a full fit."""
    X, y = _training_data
    params = dict(n_kernels=10, random_state=0)
    limit = "contract_max_n_estimators" if contracted else "n_estimators"
    if contracted:
        params["time_limit_in_minutes"] = 5
    full = Arsenal(**params, **{limit: 4})
    partial = Arsenal(**params, **{limit: 2})
    method = "fit_predict_proba" if train_estimates else "fit"
    getattr(full, method)(X, y)
    getattr(partial, method)(X[:, 0], y)
    partial.save_checkpoint(checkpoint_directory / "arsenal.pkl")
    restored = Arsenal.load_checkpoint(checkpoint_directory / "arsenal.pkl")
    restored.set_params(**{limit: 4})
    assert restored.resume_fit(X, y) is restored
    assert restored.is_fitted
    _assert_same_ensemble(full, restored, X)
    restored.set_params(**{limit: 2})
    getattr(restored, method)(X, y)
    assert restored.n_estimators_ == 2
    _assert_same_ensemble(partial, restored, X)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_arsenal_interrupted_batch(_training_data, checkpoint_directory, n_jobs):
    """A failed next batch can be recovered without skipping random seeds."""
    X, y = _training_data
    path = checkpoint_directory / "arsenal.pkl"
    params = dict(n_kernels=10, n_estimators=4, random_state=0, n_jobs=n_jobs)
    uninterrupted = Arsenal(**params).fit(X, y)
    partial = Arsenal(**params, checkpoint_path=path, checkpoint_interval=1e-12)
    from aeon.classification.convolution_based._arsenal import _run_jobs

    calls = 0

    def fail_second_batch(tasks, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            list(tasks)
            raise RuntimeError("interrupted")
        return _run_jobs(tasks, *args, **kwargs)

    with patch(
        "aeon.classification.convolution_based._arsenal._run_jobs", fail_second_batch
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit(X, y)
    restored = Arsenal.load_checkpoint(path)
    assert not restored.is_fitted
    assert restored.n_estimators_ == n_jobs
    restored.resume_fit(X, y)
    _assert_same_ensemble(uninterrupted, restored, X)
    assert Arsenal.load_checkpoint(path).is_fitted


def test_arsenal_extend_and_truncate(_training_data):
    """A completed ensemble grows to a raised limit and truncates to a lowered one."""
    X, y = _training_data
    params = dict(n_kernels=10, random_state=0)
    uninterrupted = Arsenal(**params, n_estimators=5).fit(X, y)

    # growing a completed fit reproduces an uninterrupted fit of the same size,
    # so extending an ensemble costs nothing in fidelity
    extended = Arsenal(**params, n_estimators=2).fit(X, y)
    assert extended.n_estimators_ == 2
    extended.set_params(n_estimators=5).resume_fit(X, y)
    assert extended.n_estimators_ == 5
    _assert_same_ensemble(uninterrupted, extended, X)

    # a lowered limit discards the newest members, leaving the oldest untouched
    extended.set_params(n_estimators=2).resume_fit(X, y)
    assert extended.n_estimators_ == 2
    assert len(extended.estimators_) == len(extended.weights_) == 2
    np.testing.assert_array_equal(
        extended.predict_proba(X),
        Arsenal(**params, n_estimators=2).fit(X, y).predict_proba(X),
    )


def test_arsenal_resume_validation(_training_data):
    """Wrong values, labels and model parameters fail without changing metadata."""
    X, y = _training_data
    classifier = Arsenal(n_kernels=10, n_estimators=1, random_state=0).fit(X, y)
    original = joblib_hash((classifier.classes_, classifier.metadata_))
    changed_X = X.copy()
    changed_X[0, 0, 0] += 1
    changed_y = y.copy()
    changed_y[0] = 99
    for candidate_X, candidate_y in [
        (changed_X, y),
        (X, changed_y),
        (X[::-1], y[::-1]),
    ]:
        with pytest.raises(ValueError, match="Unable to resume fitting: X and y"):
            classifier.resume_fit(candidate_X, candidate_y)
    assert joblib_hash((classifier.classes_, classifier.metadata_)) == original
    classifier.set_params(n_kernels=20)
    with pytest.raises(ValueError, match="Model-building parameters"):
        classifier.resume_fit(X, y)
    with pytest.raises(ValueError, match="call fit first"):
        Arsenal().resume_fit(X, y)
    with pytest.raises(NotImplementedError, match="does not support"):
        MockClassifier().resume_fit(X, y)


def test_arsenal_contract_budget_persists(_training_data):
    """The contract covers total training time across calls, not time per call."""
    X, y = _training_data
    params = dict(
        n_kernels=10,
        time_limit_in_minutes=60,
        contract_max_n_estimators=4,
        random_state=0,
    )
    # elapsed time is set explicitly throughout, so no assertion depends on
    # how long the ensemble actually takes to build
    uninterrupted = Arsenal(**params).fit(X, y)
    assert uninterrupted.n_estimators_ == 4

    partial = Arsenal(**params).set_params(contract_max_n_estimators=1)
    partial.fit(X, y)
    assert partial.n_estimators_ == 1
    partial.set_params(contract_max_n_estimators=4)

    # a budget already spent is not refilled by resuming
    partial.fit_elapsed_time_ = 1e9
    partial.resume_fit(X, y)
    assert partial.n_estimators_ == 1

    # budget that remains is spent by the resumed call
    partial.fit_elapsed_time_ = 0.0
    partial.resume_fit(X, y)
    assert partial.n_estimators_ == 4
    _assert_same_ensemble(uninterrupted, partial, X)

    # fit always starts the budget afresh
    partial.fit_elapsed_time_ = 1e9
    partial.fit(X, y)
    assert partial.n_estimators_ == 4


def test_arsenal_random_state_object(_training_data, checkpoint_directory):
    """A RandomState object is copied, not shared, and survives resuming."""
    X, y = _training_data
    seed = np.random.RandomState(0)
    caller_state = joblib_hash(seed.get_state())
    full = Arsenal(n_kernels=10, random_state=seed, n_estimators=4)
    partial = Arsenal(n_kernels=10, random_state=deepcopy(seed), n_estimators=2)
    full.fit_predict(X, y)
    partial.fit_predict(X, y)

    # continuation state is copied at fit, so the caller's generator is not
    # advanced and the model and OOB streams are independent of each other
    assert joblib_hash(seed.get_state()) == caller_state
    assert full._rng is not full._train_rng

    partial.save_checkpoint(checkpoint_directory / "arsenal.pkl")
    restored = Arsenal.load_checkpoint(checkpoint_directory / "arsenal.pkl")
    restored.n_estimators = 4
    before_prediction = joblib_hash(restored._rng.get_state())
    restored.predict(X)
    assert joblib_hash(restored._rng.get_state()) == before_prediction
    restored.resume_fit(X, y)
    _assert_same_ensemble(full, restored, X)


def test_arsenal_checkpoint_estimator_check():
    """The capability tag schedules the generic interruption/recovery check."""
    results = check_estimator(
        Arsenal,
        checks_to_run="check_checkpointing_classifier",
        raise_exceptions=True,
    )
    assert results and all(result == "PASSED" for result in results.values())
