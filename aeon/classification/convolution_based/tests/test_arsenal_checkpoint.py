"""Arsenal continuation tests, including parallel and interrupted training."""

from copy import deepcopy
from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash

from aeon.classification.convolution_based import Arsenal
from aeon.classification.convolution_based._arsenal import _run_jobs
from aeon.testing.data_generation import make_example_3d_numpy

N_CASES, N_TIMEPOINTS = 12, 24
N_KERNELS = 10


@pytest.fixture
def training_data():
    """Provide a small deterministic classification dataset."""
    return make_example_3d_numpy(
        n_cases=N_CASES, n_timepoints=N_TIMEPOINTS, random_state=42
    )


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
def test_arsenal_resume(training_data, tmp_path, contracted, train_estimates):
    """A resumed ensemble has the same members, RNG and OOB state as a full fit."""
    X, y = training_data
    initial_size, target_size = 2, 4
    params = dict(n_kernels=N_KERNELS, random_state=0)
    limit = "contract_max_n_estimators" if contracted else "n_estimators"
    if contracted:
        params["time_limit_in_minutes"] = 5
    full = Arsenal(**params, **{limit: target_size})
    partial = Arsenal(**params, **{limit: initial_size})
    method = "fit_predict_proba" if train_estimates else "fit"
    getattr(full, method)(X, y)
    getattr(partial, method)(X[:, 0], y)
    partial.save_checkpoint(tmp_path / "arsenal.pkl")
    restored = Arsenal.load_checkpoint(tmp_path / "arsenal.pkl")
    restored.set_params(**{limit: target_size})
    assert restored.resume_fit(X, y) is restored
    assert restored.is_fitted
    _assert_same_ensemble(full, restored, X)
    restored.set_params(**{limit: initial_size})
    getattr(restored, method)(X, y)
    assert restored.n_estimators_ == initial_size
    _assert_same_ensemble(partial, restored, X)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_arsenal_interrupted_batch(training_data, tmp_path, n_jobs):
    """A failed next batch can be recovered without skipping random seeds."""
    X, y = training_data
    path = tmp_path / "arsenal.pkl"
    params = dict(n_kernels=N_KERNELS, n_estimators=4, random_state=0, n_jobs=n_jobs)
    uninterrupted = Arsenal(**params).fit(X, y)
    checkpoint_interval = 1
    partial = Arsenal(
        **params, checkpoint_path=path, checkpoint_interval=checkpoint_interval
    )
    calls = 0

    def fail_second_batch(tasks, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            list(tasks)
            raise RuntimeError("interrupted")
        return _run_jobs(tasks, *args, **kwargs)

    with (
        patch(
            "aeon.classification.convolution_based._arsenal._run_jobs",
            fail_second_batch,
        ),
        patch("aeon.base._checkpoint.time") as clock,
    ):
        clock.monotonic.side_effect = count(start=0, step=checkpoint_interval * 60)
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit(X, y)
    restored = Arsenal.load_checkpoint(path)
    assert not restored.is_fitted
    assert restored.n_estimators_ == n_jobs
    restored.resume_fit(X, y)
    _assert_same_ensemble(uninterrupted, restored, X)
    assert Arsenal.load_checkpoint(path).is_fitted


def test_arsenal_extend_and_truncate(training_data):
    """A completed ensemble grows to a raised limit and truncates to a lowered one."""
    X, y = training_data
    initial_size, target_size = 2, 5
    params = dict(n_kernels=N_KERNELS, random_state=0)
    uninterrupted = Arsenal(**params, n_estimators=target_size).fit(X, y)

    extended = Arsenal(**params, n_estimators=initial_size).fit(X, y)
    assert extended.n_estimators_ == initial_size
    extended.set_params(n_estimators=target_size).resume_fit(X, y)
    assert extended.n_estimators_ == target_size
    _assert_same_ensemble(uninterrupted, extended, X)

    extended.set_params(n_estimators=initial_size).resume_fit(X, y)
    assert extended.n_estimators_ == initial_size
    assert len(extended.estimators_) == len(extended.weights_) == initial_size
    np.testing.assert_array_equal(
        extended.predict_proba(X),
        Arsenal(**params, n_estimators=initial_size).fit(X, y).predict_proba(X),
    )


def test_arsenal_resume_validation(training_data):
    """Wrong values, labels and model parameters fail without changing metadata."""
    X, y = training_data
    classifier = Arsenal(n_kernels=N_KERNELS, n_estimators=1, random_state=0).fit(X, y)
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
        with pytest.raises(ValueError, match="X and y"):
            classifier.resume_fit(candidate_X, candidate_y)
    assert joblib_hash((classifier.classes_, classifier.metadata_)) == original
    classifier.set_params(n_kernels=20)
    with pytest.raises(ValueError, match="parameters"):
        classifier.resume_fit(X, y)
    with pytest.raises(ValueError, match="fit first"):
        Arsenal().resume_fit(X, y)


def test_arsenal_contract_budget_persists(training_data):
    """The contract covers total training time across calls, not time per call."""
    X, y = training_data
    initial_size, target_size = 1, 4
    budget_minutes = 60
    params = dict(
        n_kernels=N_KERNELS,
        time_limit_in_minutes=budget_minutes,
        contract_max_n_estimators=initial_size,
        random_state=0,
    )
    partial = Arsenal(**params)
    partial.fit(X, y)
    assert partial.n_estimators_ == initial_size
    partial.set_params(contract_max_n_estimators=target_size)

    # a budget already spent is not refilled by resuming
    partial.fit_elapsed_time_ = budget_minutes * 60
    partial.resume_fit(X, y)
    assert partial.n_estimators_ == initial_size

    # budget that remains is spent by the resumed call
    partial.fit_elapsed_time_ = 0.0
    partial.resume_fit(X, y)
    assert partial.n_estimators_ == target_size

    # fit always starts the budget afresh
    partial.fit_elapsed_time_ = budget_minutes * 60
    partial.fit(X, y)
    assert partial.n_estimators_ == target_size


def test_arsenal_random_state_object(training_data, tmp_path):
    """A RandomState object is copied, not shared, and survives resuming."""
    X, y = training_data
    initial_size, target_size = 2, 4
    seed = np.random.RandomState(0)
    caller_state = joblib_hash(seed.get_state())
    full = Arsenal(n_kernels=N_KERNELS, random_state=seed, n_estimators=target_size)
    partial = Arsenal(
        n_kernels=N_KERNELS, random_state=deepcopy(seed), n_estimators=initial_size
    )
    full.fit_predict(X, y)
    partial.fit_predict(X, y)

    # continuation state is copied at fit, so the caller's generator is not
    # advanced and the model and OOB streams are independent of each other
    assert joblib_hash(seed.get_state()) == caller_state
    assert full._rng is not full._train_rng

    partial.save_checkpoint(tmp_path / "arsenal.pkl")
    restored = Arsenal.load_checkpoint(tmp_path / "arsenal.pkl")
    restored.n_estimators = target_size
    before_prediction = joblib_hash(restored._rng.get_state())
    restored.predict(X)
    assert joblib_hash(restored._rng.get_state()) == before_prediction
    restored.resume_fit(X, y)
    _assert_same_ensemble(full, restored, X)


def test_arsenal_resume_fit_predict_proba(training_data, tmp_path):
    """Train estimates from a resumed fit equal those of an uninterrupted fit."""
    X, y = training_data
    initial_size, target_size = 2, 4
    params = dict(n_kernels=N_KERNELS, random_state=0)
    full = Arsenal(**params, n_estimators=target_size)
    expected = full.fit_predict_proba(X, y)
    partial = Arsenal(**params, n_estimators=initial_size)
    partial.fit_predict_proba(X, y)
    partial.save_checkpoint(tmp_path / "arsenal.pkl")
    restored = Arsenal.load_checkpoint(tmp_path / "arsenal.pkl")
    restored.set_params(n_estimators=target_size)

    np.testing.assert_allclose(restored.resume_fit_predict_proba(X, y), expected)
    assert restored.is_fitted
    np.testing.assert_allclose(restored.predict_proba(X), full.predict_proba(X))


def test_arsenal_resume_fit_predict_proba_needs_train_estimate_fit(training_data):
    """A fit begun with fit has no train estimates to continue."""
    X, y = training_data
    classifier = Arsenal(n_kernels=N_KERNELS, n_estimators=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="train estimates"):
        classifier.resume_fit_predict_proba(X, y)
