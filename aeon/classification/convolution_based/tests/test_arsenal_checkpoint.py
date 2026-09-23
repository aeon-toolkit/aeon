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
def training_data():
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


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("contracted", [False, True])
@pytest.mark.parametrize("train_estimates", [False, True])
def test_arsenal_resume(
    training_data, checkpoint_directory, n_jobs, contracted, train_estimates
):
    """A resumed ensemble has the same members, RNG and OOB state as a full fit."""
    X, y = training_data
    params = dict(n_kernels=10, random_state=0, n_jobs=n_jobs)
    limit = "contract_max_n_estimators" if contracted else "n_estimators"
    if contracted:
        params["time_limit_in_minutes"] = 5
    full = Arsenal(**params, **{limit: 4})
    partial = Arsenal(**params, **{limit: 2})
    method = "fit_predict_proba" if train_estimates else "fit"
    getattr(full, method)(X, y)
    getattr(partial, method)(X, y)
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
def test_arsenal_interrupted_batch(training_data, checkpoint_directory, n_jobs):
    """A failed next batch can be recovered without skipping random seeds."""
    X, y = training_data
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


def test_arsenal_resume_validation(training_data):
    """Wrong values, labels and model parameters fail without changing metadata."""
    X, y = training_data
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


def test_arsenal_resume_converted_data(training_data):
    """Canonical preprocessing permits equivalent 2D and 3D input."""
    X, y = training_data
    classifier = Arsenal(n_kernels=10, n_estimators=1, random_state=0).fit(X[:, 0], y)
    classifier.n_estimators = 2
    classifier.resume_fit(X, y)
    assert classifier.n_estimators_ == 2


def test_arsenal_new_contract_budget(training_data, checkpoint_directory):
    """Each call receives a new time budget regardless of accumulated time."""
    X, y = training_data
    classifier = Arsenal(
        n_kernels=10,
        time_limit_in_minutes=1e-12,
        contract_max_n_estimators=4,
        random_state=0,
    ).fit(X, y)
    assert classifier.n_estimators_ == 1
    classifier.fit_elapsed_time_ = 1e9
    classifier.save_checkpoint(checkpoint_directory / "arsenal.pkl")
    restored = Arsenal.load_checkpoint(checkpoint_directory / "arsenal.pkl")
    restored.resume_fit(X, y)
    assert restored.n_estimators_ == 2
    assert restored.fit_elapsed_time_ >= 1e9


@pytest.mark.parametrize("method", ["fit", "fit_predict", "fit_predict_proba"])
def test_arsenal_random_state_object(training_data, checkpoint_directory, method):
    """Shared model/OOB RNGs remain shared after loading and resuming."""
    X, y = training_data
    params = dict(n_kernels=10, random_state=np.random.RandomState(0))
    full = Arsenal(**deepcopy(params), n_estimators=4)
    partial = Arsenal(**deepcopy(params), n_estimators=2)
    getattr(full, method)(X, y)
    getattr(partial, method)(X, y)
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
