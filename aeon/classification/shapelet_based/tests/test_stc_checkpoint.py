"""STC continuation tests, covering both phases of the build."""

from contextlib import contextmanager
from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash
from sklearn.ensemble import RandomForestClassifier

from aeon.base._estimators.sklearn._rotation_forest import _run_jobs
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

N_CASES, N_TIMEPOINTS = 12, 24
N_FOREST_TREES = 4
CHECKPOINT_INTERVAL = 1
_PARAMS = dict(n_shapelet_samples=10, max_shapelets=3, batch_size=5, random_state=0)


@pytest.fixture
def training_data():
    """Provide a small deterministic classification dataset."""
    return make_example_3d_numpy(
        n_cases=N_CASES, n_timepoints=N_TIMEPOINTS, random_state=42
    )


def _assert_same_pipeline(expected, actual, X):
    np.testing.assert_array_equal(actual.predict_proba(X), expected.predict_proba(X))
    assert joblib_hash(actual.transformer_.shapelets) == joblib_hash(
        expected.transformer_.shapelets
    )
    np.testing.assert_array_equal(
        actual._transformed_train_data, expected._transformed_train_data
    )


def _fail_forest_batch(failing_call):
    """Patch the forest so its ``failing_call``-th batch of trees raises."""
    calls = 0

    def run_jobs(tasks, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failing_call:
            list(tasks)
            raise RuntimeError("interrupted")
        return _run_jobs(tasks, *args, **kwargs)

    return patch("aeon.base._estimators.sklearn._rotation_forest._run_jobs", run_jobs)


@contextmanager
def _periodic_checkpoints():
    """Make every safe boundary due for a periodic checkpoint."""
    with patch("aeon.base._checkpoint.time") as clock:
        clock.monotonic.side_effect = count(start=0, step=CHECKPOINT_INTERVAL * 60)
        yield


def _transform_not_rerun():
    """Fail if the shapelet transform is fitted again."""
    return patch.object(
        RandomShapeletTransform,
        "fit_transform",
        side_effect=AssertionError("the transform was rerun"),
    )


def _forest_stc(**kwargs):
    return ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=N_FOREST_TREES),
        **_PARAMS,
        **kwargs,
    )


def test_stc_resume_continues_the_forest(training_data, tmp_path):
    """An interrupted forest resumes into the same pipeline as a full fit."""
    X, y = training_data
    path = tmp_path / "stc.pkl"
    uninterrupted = _forest_stc().fit(X, y)
    partial = _forest_stc(checkpoint_path=path, checkpoint_interval=CHECKPOINT_INTERVAL)
    with _fail_forest_batch(3), _periodic_checkpoints():
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit(X, y)

    restored = ShapeletTransformClassifier.load_checkpoint(path)
    assert not restored.is_fitted
    assert 0 < restored.estimator_._n_estimators < N_FOREST_TREES
    with _transform_not_rerun():
        assert restored.resume_fit(X, y) is restored
    assert restored.is_fitted
    assert restored.estimator_._n_estimators == N_FOREST_TREES
    _assert_same_pipeline(uninterrupted, restored, X)


def test_stc_checkpoints_when_the_transform_completes(training_data, tmp_path):
    """The transform boundary is saved even with no interval configured."""
    X, y = training_data
    path = tmp_path / "stc.pkl"
    classifier = _forest_stc(checkpoint_path=path)
    with patch.object(
        RotationForestClassifier, "fit", side_effect=RuntimeError("interrupted")
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            classifier.fit(X, y)

    restored = ShapeletTransformClassifier.load_checkpoint(path)
    assert restored._transformed_train_data is not None
    assert restored.fit_elapsed_time_ > 0
    assert not hasattr(restored.estimator_, "estimators_")
    with _transform_not_rerun():
        restored.resume_fit(X, y)
    assert restored.is_fitted
    _assert_same_pipeline(_forest_stc().fit(X, y), restored, X)


def test_stc_non_checkpointable_estimator(training_data, tmp_path):
    """An estimator that cannot resume is refitted on the saved transform."""
    X, y = training_data
    path = tmp_path / "stc.pkl"
    params = dict(
        estimator=RandomForestClassifier(n_estimators=N_FOREST_TREES), **_PARAMS
    )
    uninterrupted = ShapeletTransformClassifier(**params).fit(X, y)
    classifier = ShapeletTransformClassifier(**params, checkpoint_path=path)
    with patch.object(
        RandomForestClassifier, "fit", side_effect=RuntimeError("interrupted")
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            classifier.fit(X, y)

    restored = ShapeletTransformClassifier.load_checkpoint(path)
    with _transform_not_rerun():
        restored.resume_fit(X, y)
    _assert_same_pipeline(uninterrupted, restored, X)


def test_stc_resume_after_completion(training_data):
    """Resuming a completed fit is a no-op that leaves the pipeline alone."""
    X, y = training_data
    classifier = _forest_stc().fit(X, y)
    expected = classifier.predict_proba(X)
    classifier.resume_fit(X, y)
    assert classifier.is_fitted
    np.testing.assert_array_equal(classifier.predict_proba(X), expected)


def test_stc_resume_validation(training_data):
    """Wrong values, labels and model parameters are refused."""
    X, y = training_data
    classifier = _forest_stc().fit(X, y)
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

    classifier.set_params(n_shapelet_samples=20)
    with pytest.raises(ValueError, match="parameters"):
        classifier.resume_fit(X, y)
    classifier.set_params(**_PARAMS)
    with pytest.raises(ValueError, match="fit first"):
        _forest_stc().resume_fit(X, y)


def test_stc_random_state_object(training_data, tmp_path):
    """A RandomState instance shared with the components can still resume."""
    X, y = training_data
    path = tmp_path / "stc.pkl"
    params = dict(_PARAMS)
    del params["random_state"]
    uninterrupted = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=N_FOREST_TREES),
        random_state=np.random.RandomState(0),
        **params,
    ).fit(X, y)

    classifier = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=N_FOREST_TREES),
        random_state=np.random.RandomState(0),
        checkpoint_path=path,
        **params,
    )
    with patch.object(
        RotationForestClassifier, "fit", side_effect=RuntimeError("interrupted")
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            classifier.fit(X, y)

    restored = ShapeletTransformClassifier.load_checkpoint(path)
    restored.resume_fit(X, y)
    _assert_same_pipeline(uninterrupted, restored, X)


def test_stc_resume_applies_runtime_settings(training_data):
    """Settings that may change between calls reach the estimator on resuming."""
    X, y = training_data
    classifier = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(
            n_estimators=N_FOREST_TREES, contract_max_n_estimators=N_FOREST_TREES
        ),
        n_jobs=1,
        **_PARAMS,
    ).fit(X, y)
    classifier.set_params(n_jobs=2, time_limit_in_minutes=6)
    classifier.resume_fit(X, y)
    assert classifier.estimator_.n_jobs == 2
    # the estimator receives its third of the overall contract
    assert classifier.estimator_.time_limit_in_minutes == pytest.approx(2)


@pytest.mark.parametrize("interrupted_in", ["estimator", "forest"])
def test_stc_resume_fit_predict_proba(training_data, tmp_path, interrupted_in):
    """Train estimates resumed after either phase equal an uninterrupted fit's."""
    X, y = training_data
    path = tmp_path / "stc.pkl"
    expected = _forest_stc().fit_predict_proba(X, y)
    partial = _forest_stc(checkpoint_path=path, checkpoint_interval=CHECKPOINT_INTERVAL)
    if interrupted_in == "estimator":
        interruption = patch.object(
            RotationForestClassifier,
            "fit_predict_proba",
            side_effect=RuntimeError("interrupted"),
        )
    else:
        interruption = _fail_forest_batch(2)
    with interruption, _periodic_checkpoints():
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit_predict_proba(X, y)

    restored = ShapeletTransformClassifier.load_checkpoint(path)
    if interrupted_in == "estimator":
        assert not hasattr(restored.estimator_, "estimators_")
    else:
        assert 0 < restored.estimator_._n_estimators < N_FOREST_TREES
    with _transform_not_rerun():
        train = restored.resume_fit_predict_proba(X, y)
    assert restored.is_fitted
    np.testing.assert_allclose(train, expected)


def test_stc_resume_fit_predict_proba_needs_train_estimate_fit(training_data):
    """A fit begun with fit has no train estimates to continue."""
    X, y = training_data
    classifier = _forest_stc().fit(X, y)
    with pytest.raises(ValueError, match="train estimates"):
        classifier.resume_fit_predict_proba(X, y)


def test_stc_resume_without_contract_restores_estimator_limit(training_data):
    """Resuming with no contract gives the estimator its own time limit again."""
    X, y = training_data
    estimator = RotationForestClassifier(n_estimators=N_FOREST_TREES)
    classifier = ShapeletTransformClassifier(
        estimator=estimator, time_limit_in_minutes=3, **_PARAMS
    ).fit(X, y)
    classifier.set_params(time_limit_in_minutes=0).resume_fit(X, y)
    assert (
        classifier.estimator_.time_limit_in_minutes == estimator.time_limit_in_minutes
    )
