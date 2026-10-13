"""Rotation Forest continuation tests, for the classifier and the regressor."""

from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash

from aeon.classification.sklearn import RotationForestClassifier
from aeon.regression.sklearn import RotationForestRegressor
from aeon.testing.data_generation import make_example_2d_numpy_collection

_FORESTS = [RotationForestClassifier, RotationForestRegressor]


@pytest.fixture
def _training_data():
    """Provide a small deterministic tabular dataset."""
    X, y = make_example_2d_numpy_collection(
        n_cases=20, n_timepoints=12, random_state=42
    )
    return X, y


def _targets(forest_class, y):
    """Class labels for the classifier, continuous values for the regressor."""
    return y if forest_class is RotationForestClassifier else y.astype(float) + 0.5


def _assert_same_forest(expected, actual, X):
    assert actual._n_estimators == expected._n_estimators
    assert (
        len(actual.estimators_)
        == len(actual._pcas)
        == len(actual._groups)
        == actual._n_estimators
    )
    np.testing.assert_array_equal(actual.predict(X), expected.predict(X))
    for expected_groups, actual_groups in zip(expected._groups, actual._groups):
        assert joblib_hash(expected_groups) == joblib_hash(actual_groups)
    assert joblib_hash(actual._rng.get_state()) == joblib_hash(
        expected._rng.get_state()
    )


@pytest.mark.parametrize("forest_class", _FORESTS)
@pytest.mark.parametrize("contracted", [False, True])
def test_rotation_forest_resume(
    _training_data, checkpoint_directory, forest_class, contracted
):
    """A resumed forest holds the same trees and RNG state as a full fit."""
    X, y = _training_data
    y = _targets(forest_class, y)
    initial_size, target_size = 2, 5
    params = dict(random_state=0)
    limit = "contract_max_n_estimators" if contracted else "n_estimators"
    if contracted:
        params["time_limit_in_minutes"] = 5
    full = forest_class(**params, **{limit: target_size}).fit(X, y)
    partial = forest_class(**params, **{limit: initial_size}).fit(X, y)
    assert partial._n_estimators == initial_size

    partial.save_checkpoint(checkpoint_directory / "rotf.pkl")
    restored = forest_class.load_checkpoint(checkpoint_directory / "rotf.pkl")
    restored.set_params(**{limit: target_size})
    assert restored.resume_fit(X, y) is restored
    _assert_same_forest(full, restored, X)

    # tree limits cover the whole forest, so a lowered one discards the newest
    # and keeps the oldest. The generator is not rewound, which is why only the
    # trees are compared here and not the random state
    restored.set_params(**{limit: initial_size}).resume_fit(X, y)
    assert restored._n_estimators == initial_size
    np.testing.assert_array_equal(restored.predict(X), partial.predict(X))
    for expected_groups, actual_groups in zip(partial._groups, restored._groups):
        assert joblib_hash(expected_groups) == joblib_hash(actual_groups)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_rotation_forest_interrupted_batch(
    _training_data, checkpoint_directory, n_jobs
):
    """A failed batch can be recovered without skipping tree seeds."""
    X, y = _training_data
    path = checkpoint_directory / "rotf.pkl"
    params = dict(n_estimators=4, random_state=0, n_jobs=n_jobs)
    uninterrupted = RotationForestClassifier(**params).fit(X, y)
    checkpoint_interval = 1
    partial = RotationForestClassifier(
        **params, checkpoint_path=path, checkpoint_interval=checkpoint_interval
    )
    from aeon.base._estimators.sklearn._rotation_forest import _run_jobs

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
            "aeon.base._estimators.sklearn._rotation_forest._run_jobs",
            fail_second_batch,
        ),
        patch("aeon.base._checkpoint.time") as clock,
    ):
        clock.monotonic.side_effect = count(start=0, step=checkpoint_interval * 60)
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit(X, y)

    restored = RotationForestClassifier.load_checkpoint(path)
    assert restored._n_estimators == n_jobs
    restored.resume_fit(X, y)
    _assert_same_forest(uninterrupted, restored, X)


def test_rotation_forest_resume_keeps_oob_state(_training_data):
    """Out of bag state from fit_predict survives and grows with the forest."""
    X, y = _training_data
    initial_size, target_size = 2, 5
    params = dict(random_state=0)
    full = RotationForestClassifier(**params, n_estimators=target_size)
    partial = RotationForestClassifier(**params, n_estimators=initial_size)
    expected = full.fit_predict_proba(X, y)
    partial.fit_predict_proba(X, y)
    assert len(partial._transformed_data) == initial_size

    partial.set_params(n_estimators=target_size).resume_fit(X, y)
    assert len(partial._transformed_data) == target_size
    _assert_same_forest(full, partial, X)
    for expected_data, actual_data in zip(
        full._transformed_data, partial._transformed_data
    ):
        np.testing.assert_array_equal(expected_data, actual_data)

    # the estimates themselves are not returned by resume_fit, but the state
    # they are built from is complete, so recomputing them reproduces the fit
    np.testing.assert_array_equal(expected, full._fit_predict_rotf(X, y))


@pytest.mark.parametrize("forest_class", _FORESTS)
def test_rotation_forest_resume_validation(_training_data, forest_class):
    """Wrong values, targets and model parameters are refused."""
    X, y = _training_data
    y = _targets(forest_class, y)
    forest = forest_class(n_estimators=1, random_state=0).fit(X, y)
    changed_X = X.copy()
    changed_X[0, 0] += 1
    for candidate_X, candidate_y in [
        (changed_X, y),
        (X[::-1], y[::-1]),
    ]:
        with pytest.raises(ValueError, match="Unable to resume fitting: X and y"):
            forest.resume_fit(candidate_X, candidate_y)

    forest.set_params(min_group=2)
    with pytest.raises(ValueError, match="Model-building parameters"):
        forest.resume_fit(X, y)
    forest.set_params(min_group=3, n_estimators=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        forest.resume_fit(X, y)
    with pytest.raises(ValueError, match="call fit first"):
        forest_class().resume_fit(X, y)


def test_rotation_forest_contract_budget_persists(_training_data):
    """The contract covers total training time across calls, not time per call."""
    X, y = _training_data
    initial_size, target_size = 1, 4
    budget_minutes = 60
    params = dict(
        time_limit_in_minutes=budget_minutes,
        contract_max_n_estimators=initial_size,
        random_state=0,
    )
    # elapsed time is set explicitly throughout, so no assertion depends on
    # how long the forest actually takes to build
    partial = RotationForestClassifier(**params).fit(X, y)
    assert partial._n_estimators == initial_size
    partial.set_params(contract_max_n_estimators=target_size)

    partial.fit_elapsed_time_ = budget_minutes * 60
    partial.resume_fit(X, y)
    assert partial._n_estimators == initial_size

    partial.fit_elapsed_time_ = 0.0
    partial.resume_fit(X, y)
    assert partial._n_estimators == target_size

    # fit always starts the budget afresh
    partial.fit_elapsed_time_ = budget_minutes * 60
    partial.fit(X, y)
    assert partial._n_estimators == target_size


def test_rotation_forest_single_class(_training_data):
    """A forest with nothing to build is complete and resuming changes nothing."""
    X, y = _training_data
    y = np.zeros_like(y)
    forest = RotationForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    assert not hasattr(forest, "estimators_")
    forest.set_params(n_estimators=5).resume_fit(X, y)
    assert not hasattr(forest, "estimators_")
    np.testing.assert_array_equal(forest.predict(X), y)
