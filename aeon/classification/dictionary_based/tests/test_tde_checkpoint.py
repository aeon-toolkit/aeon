"""TDE continuation tests, including interrupted and contracted training."""

from copy import deepcopy
from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash

from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.testing.mock_estimators import MockClassifier

_PARAMS = dict(max_ensemble_size=2, randomly_selected_params=3, random_state=0)


@pytest.fixture
def _training_data():
    """Provide a small deterministic classification dataset."""
    return make_example_3d_numpy(n_cases=12, n_timepoints=24, random_state=42)


def _assert_same_ensemble(expected, actual, X):
    # the search is driven by the evaluation count, the retained members and the
    # history feeding the guided selection, so all three must match, not just
    # the predictions they happen to produce
    assert actual._num_classifiers == expected._num_classifiers
    assert actual.n_estimators_ == expected.n_estimators_
    assert len(actual.estimators_) == len(actual.weights_) == actual.n_estimators_
    np.testing.assert_array_equal(actual.weights_, expected.weights_)
    np.testing.assert_array_equal(actual.predict_proba(X), expected.predict_proba(X))
    assert actual._lowest_acc == expected._lowest_acc
    assert actual._lowest_acc_idx == expected._lowest_acc_idx
    assert actual._prev_parameters_x == expected._prev_parameters_x
    assert actual._prev_parameters_y == expected._prev_parameters_y
    assert actual._possible_parameters == expected._possible_parameters
    np.testing.assert_array_equal(
        actual._candidate_parameters, expected._candidate_parameters
    )
    assert joblib_hash(actual._rng.get_state()) == joblib_hash(
        expected._rng.get_state()
    )
    for expected_member, actual_member in zip(expected.estimators_, actual.estimators_):
        assert joblib_hash(expected_member._transformed_data) == joblib_hash(
            actual_member._transformed_data
        )
        np.testing.assert_array_equal(
            expected_member._subsample, actual_member._subsample
        )


@pytest.mark.parametrize("contracted, train_estimates", [(False, False), (True, True)])
def test_tde_resume(_training_data, checkpoint_directory, contracted, train_estimates):
    """A resumed search evaluates the same candidates as an uninterrupted fit."""
    X, y = _training_data
    initial_samples, target_samples = 2, 5
    params = dict(_PARAMS)
    limit = "contract_max_n_parameter_samples" if contracted else "n_parameter_samples"
    if contracted:
        params["time_limit_in_minutes"] = 5
    full = TemporalDictionaryEnsemble(**params, **{limit: target_samples})
    partial = TemporalDictionaryEnsemble(**params, **{limit: initial_samples})
    method = "fit_predict_proba" if train_estimates else "fit"
    getattr(full, method)(X, y)
    getattr(partial, method)(X, y)
    assert partial._num_classifiers == initial_samples

    partial.save_checkpoint(checkpoint_directory / "tde.pkl")
    restored = TemporalDictionaryEnsemble.load_checkpoint(
        checkpoint_directory / "tde.pkl"
    )
    restored.set_params(**{limit: target_samples})
    assert restored.resume_fit(X, y) is restored
    assert restored.is_fitted
    _assert_same_ensemble(full, restored, X)
    if train_estimates:
        # continuing a fit begun with fit_predict_proba keeps collecting the
        # member train predictions the estimates are built from
        for member in restored.estimators_:
            assert len(member._train_predictions) == len(member._subsample)

    # a budget already spent is not topped up, and fit always starts afresh
    restored.resume_fit(X, y)
    _assert_same_ensemble(full, restored, X)
    restored.set_params(**{limit: initial_samples})
    getattr(restored, method)(X, y)
    assert restored._num_classifiers == initial_samples
    _assert_same_ensemble(partial, restored, X)


def test_tde_interrupted_candidate(_training_data, checkpoint_directory):
    """A failed candidate evaluation is repeated exactly, not skipped."""
    X, y = _training_data
    path = checkpoint_directory / "tde.pkl"
    n_parameter_samples = 5
    uninterrupted = TemporalDictionaryEnsemble(
        n_parameter_samples=n_parameter_samples, **_PARAMS
    ).fit(X, y)
    checkpoint_interval = 1
    partial = TemporalDictionaryEnsemble(
        n_parameter_samples=n_parameter_samples,
        checkpoint_path=path,
        checkpoint_interval=checkpoint_interval,
        **_PARAMS,
    )
    original_train_acc = TemporalDictionaryEnsemble._individual_train_acc
    calls = 0

    def fail_fourth_candidate(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("interrupted")
        return original_train_acc(self, *args, **kwargs)

    with (
        patch.object(
            TemporalDictionaryEnsemble,
            "_individual_train_acc",
            fail_fourth_candidate,
        ),
        patch("aeon.base._checkpoint.time") as clock,
    ):
        clock.monotonic.side_effect = count(start=0, step=checkpoint_interval * 60)
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit(X, y)

    # the failed candidate left no trace, in memory or on disk, so both the
    # interrupted object and the checkpoint can be resumed
    assert partial._num_classifiers == 3
    restored = TemporalDictionaryEnsemble.load_checkpoint(path)
    assert not restored.is_fitted
    assert restored._num_classifiers == 3
    for recovered in (partial, restored):
        recovered.resume_fit(X, y)
        _assert_same_ensemble(uninterrupted, recovered, X)
    assert TemporalDictionaryEnsemble.load_checkpoint(path).is_fitted


def test_tde_resume_replaces_members(_training_data):
    """Members are still replaced by better candidates after resuming."""
    X, y = _training_data
    # the ensemble is full after two candidates, so the later ones can only
    # enter by replacing a retained member
    full = TemporalDictionaryEnsemble(n_parameter_samples=6, **_PARAMS).fit(X, y)
    partial = TemporalDictionaryEnsemble(n_parameter_samples=2, **_PARAMS).fit(X, y)
    assert partial.n_estimators_ == _PARAMS["max_ensemble_size"]
    partial.set_params(n_parameter_samples=6).resume_fit(X, y)
    _assert_same_ensemble(full, partial, X)
    assert partial.n_estimators_ == _PARAMS["max_ensemble_size"]
    assert partial._num_classifiers == 6
    # the accuracies retained are the best seen, whenever they were evaluated
    assert (
        sorted(member._accuracy for member in partial.estimators_)
        == sorted(partial._prev_parameters_y)[-partial.n_estimators_ :]
    )


def test_tde_resume_exhausted_parameter_space(_training_data):
    """Resuming stops cleanly when no parameter combinations are left."""
    X, y = _training_data
    combinations = [[10, 8, True, 1, True], [12, 8, False, 1, False]]
    with patch.object(
        TemporalDictionaryEnsemble,
        "_unique_parameters",
        lambda self, max_window, win_inc: deepcopy(combinations),
    ):
        classifier = TemporalDictionaryEnsemble(n_parameter_samples=10, **_PARAMS)
        classifier.fit(X, y)
        assert classifier._num_classifiers == len(combinations)
        assert classifier._possible_parameters == []
        classifier.resume_fit(X, y)
    assert classifier._num_classifiers == len(combinations)
    assert classifier.is_fitted


def test_tde_resume_validation(_training_data):
    """Wrong values, labels and model parameters fail without changing metadata."""
    X, y = _training_data
    classifier = TemporalDictionaryEnsemble(n_parameter_samples=1, **_PARAMS).fit(X, y)
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

    # retention decisions were made against max_ensemble_size as candidates were
    # evaluated, so unlike the search budget it cannot change between calls
    for changed in [{"max_ensemble_size": 5}, {"min_window": 8}]:
        classifier.set_params(**changed)
        with pytest.raises(ValueError, match="Model-building parameters"):
            classifier.resume_fit(X, y)
        classifier.set_params(**{name: _PARAMS.get(name, 10) for name in changed})

    classifier.set_params(n_parameter_samples=0)
    with pytest.raises(ValueError, match="must be positive"):
        classifier.resume_fit(X, y)
    with pytest.raises(ValueError, match="call fit first"):
        TemporalDictionaryEnsemble().resume_fit(X, y)
    with pytest.raises(NotImplementedError, match="does not support"):
        MockClassifier().resume_fit(X, y)


def test_tde_contract_budget_persists(_training_data):
    """The contract covers total training time across calls, not time per call."""
    X, y = _training_data
    initial_samples, target_samples = 1, 4
    budget_minutes = 60
    params = dict(
        time_limit_in_minutes=budget_minutes,
        contract_max_n_parameter_samples=initial_samples,
        **_PARAMS,
    )
    # elapsed time is set explicitly throughout, so no assertion depends on
    # how long the ensemble actually takes to build
    partial = TemporalDictionaryEnsemble(**params)
    partial.fit(X, y)
    assert partial._num_classifiers == initial_samples
    partial.set_params(contract_max_n_parameter_samples=target_samples)

    # a budget already spent is not refilled by resuming
    partial.fit_elapsed_time_ = budget_minutes * 60
    partial.resume_fit(X, y)
    assert partial._num_classifiers == initial_samples

    # budget that remains is spent by the resumed call
    partial.fit_elapsed_time_ = 0.0
    partial.resume_fit(X, y)
    assert partial._num_classifiers == target_samples

    # fit always starts the budget afresh
    partial.fit_elapsed_time_ = budget_minutes * 60
    partial.fit(X, y)
    assert partial._num_classifiers == target_samples


def test_tde_random_state_object(_training_data, checkpoint_directory):
    """A RandomState object is copied, not shared, and survives resuming."""
    X, y = _training_data
    initial_samples, target_samples = 2, 5
    params = dict(_PARAMS)
    del params["random_state"]
    seed = np.random.RandomState(0)
    caller_state = joblib_hash(seed.get_state())
    full = TemporalDictionaryEnsemble(
        n_parameter_samples=target_samples, random_state=seed, **params
    )
    partial = TemporalDictionaryEnsemble(
        n_parameter_samples=initial_samples, random_state=deepcopy(seed), **params
    )
    full.fit(X, y)
    partial.fit(X, y)

    # continuation state is copied at fit, so the search stream is unaffected by
    # the members, which share the caller's generator through random_state
    assert joblib_hash(seed.get_state()) == caller_state

    partial.save_checkpoint(checkpoint_directory / "tde.pkl")
    restored = TemporalDictionaryEnsemble.load_checkpoint(
        checkpoint_directory / "tde.pkl"
    )
    restored.n_parameter_samples = target_samples

    # members share the caller's generator through random_state and consume it
    # on nearest neighbour ties, so the search stream must be resumed before
    # anything else uses it, predictions included
    restored.resume_fit(X, y)
    _assert_same_ensemble(full, restored, X)

    before_prediction = joblib_hash(restored._rng.get_state())
    restored.predict(X)
    assert joblib_hash(restored._rng.get_state()) == before_prediction
