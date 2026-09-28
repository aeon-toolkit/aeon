"""STC continuation tests, covering both phases of the build."""

from itertools import count
from unittest.mock import patch

import numpy as np
import pytest
from joblib import hash as joblib_hash
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

_PARAMS = dict(n_shapelet_samples=10, max_shapelets=3, batch_size=5, random_state=0)


@pytest.fixture
def _training_data():
    """Provide a small deterministic classification dataset."""
    return make_example_3d_numpy(n_cases=12, n_timepoints=24, random_state=42)


def _assert_same_pipeline(expected, actual, X):
    np.testing.assert_array_equal(actual.predict_proba(X), expected.predict_proba(X))
    assert joblib_hash(actual.transformer_.shapelets) == joblib_hash(
        expected.transformer_.shapelets
    )
    np.testing.assert_array_equal(
        actual._transformed_train_data, expected._transformed_train_data
    )


def _forest_stc(**kwargs):
    return ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=4), **_PARAMS, **kwargs
    )


def test_stc_resume_continues_the_forest(_training_data, checkpoint_directory):
    """An interrupted forest resumes into the same pipeline as a full fit."""
    X, y = _training_data
    path = checkpoint_directory / "stc.pkl"
    uninterrupted = _forest_stc().fit(X, y)
    checkpoint_interval = 1
    partial = _forest_stc(checkpoint_path=path, checkpoint_interval=checkpoint_interval)
    from aeon.base._estimators.sklearn._rotation_forest import _run_jobs

    calls = 0

    def fail_third_batch(tasks, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            list(tasks)
            raise RuntimeError("interrupted")
        return _run_jobs(tasks, *args, **kwargs)

    with (
        patch(
            "aeon.base._estimators.sklearn._rotation_forest._run_jobs",
            fail_third_batch,
        ),
        patch("aeon.base._checkpoint.time") as clock,
    ):
        clock.monotonic.side_effect = count(start=0, step=checkpoint_interval * 60)
        with pytest.raises(RuntimeError, match="interrupted"):
            partial.fit(X, y)

    # the forest was part built when the job died, and the checkpoint holds it
    restored = ShapeletTransformClassifier.load_checkpoint(path)
    assert not restored.is_fitted
    assert 0 < restored.estimator_._n_estimators < 4

    # the transform is the expensive phase and must not be repeated
    with patch.object(
        RandomShapeletTransform,
        "fit_transform",
        side_effect=AssertionError("the transform was rerun"),
    ):
        assert restored.resume_fit(X, y) is restored
    assert restored.is_fitted
    assert restored.estimator_._n_estimators == 4
    _assert_same_pipeline(uninterrupted, restored, X)


def test_stc_checkpoints_when_the_transform_completes(
    _training_data, checkpoint_directory
):
    """The transform boundary is saved even with no interval configured."""
    X, y = _training_data
    path = checkpoint_directory / "stc.pkl"
    classifier = _forest_stc(checkpoint_path=path)
    with patch.object(
        RotationForestClassifier, "fit", side_effect=RuntimeError("interrupted")
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            classifier.fit(X, y)

    # nothing of the estimator survives, but the transform does
    restored = ShapeletTransformClassifier.load_checkpoint(path)
    assert restored._transformed_train_data is not None
    assert not hasattr(restored.estimator_, "estimators_")
    with patch.object(
        RandomShapeletTransform,
        "fit_transform",
        side_effect=AssertionError("the transform was rerun"),
    ):
        restored.resume_fit(X, y)
    assert restored.is_fitted
    _assert_same_pipeline(_forest_stc().fit(X, y), restored, X)


def test_stc_non_checkpointable_estimator(_training_data, checkpoint_directory):
    """An estimator that cannot resume is refitted on the saved transform."""
    X, y = _training_data
    path = checkpoint_directory / "stc.pkl"
    params = dict(estimator=RandomForestClassifier(n_estimators=4), **_PARAMS)
    uninterrupted = ShapeletTransformClassifier(**params).fit(X, y)
    classifier = ShapeletTransformClassifier(**params, checkpoint_path=path)
    with patch.object(
        RandomForestClassifier, "fit", side_effect=RuntimeError("interrupted")
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            classifier.fit(X, y)

    restored = ShapeletTransformClassifier.load_checkpoint(path)
    with patch.object(
        RandomShapeletTransform,
        "fit_transform",
        side_effect=AssertionError("the transform was rerun"),
    ):
        restored.resume_fit(X, y)
    _assert_same_pipeline(uninterrupted, restored, X)


def test_stc_resume_after_completion(_training_data):
    """Resuming a completed fit is a no-op that leaves the pipeline alone."""
    X, y = _training_data
    classifier = _forest_stc().fit(X, y)
    expected = classifier.predict_proba(X)
    classifier.resume_fit(X, y)
    assert classifier.is_fitted
    np.testing.assert_array_equal(classifier.predict_proba(X), expected)


def test_stc_resume_validation(_training_data):
    """Wrong values, labels and model parameters are refused."""
    X, y = _training_data
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
        with pytest.raises(ValueError, match="Unable to resume fitting: X and y"):
            classifier.resume_fit(candidate_X, candidate_y)
    assert joblib_hash((classifier.classes_, classifier.metadata_)) == original

    # the transform is complete and fixed, so what shapes it cannot change
    classifier.set_params(n_shapelet_samples=20)
    with pytest.raises(ValueError, match="Model-building parameters"):
        classifier.resume_fit(X, y)
    classifier.set_params(**_PARAMS)
    with pytest.raises(ValueError, match="call fit first"):
        _forest_stc().resume_fit(X, y)


def test_stc_random_state_object(_training_data, checkpoint_directory):
    """A RandomState instance shared with the components can still resume."""
    X, y = _training_data
    path = checkpoint_directory / "stc.pkl"
    params = dict(_PARAMS)
    del params["random_state"]
    uninterrupted = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=4),
        random_state=np.random.RandomState(0),
        **params,
    ).fit(X, y)

    classifier = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=4),
        random_state=np.random.RandomState(0),
        checkpoint_path=path,
        **params,
    )
    with patch.object(
        RotationForestClassifier, "fit", side_effect=RuntimeError("interrupted")
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            classifier.fit(X, y)

    # the transform and the cloned estimator both draw from the generator the
    # caller passed, so its state moves during fit; that is not a change of
    # model configuration and must not block continuing
    restored = ShapeletTransformClassifier.load_checkpoint(path)
    restored.resume_fit(X, y)
    _assert_same_pipeline(uninterrupted, restored, X)


def test_stc_resume_applies_runtime_settings(_training_data):
    """Settings that may change between calls reach the estimator on resuming."""
    X, y = _training_data
    classifier = ShapeletTransformClassifier(
        estimator=RotationForestClassifier(n_estimators=4, contract_max_n_estimators=4),
        n_jobs=1,
        **_PARAMS,
    ).fit(X, y)
    classifier.set_params(n_jobs=2, time_limit_in_minutes=6)
    classifier.resume_fit(X, y)
    assert classifier.estimator_.n_jobs == 2
    # the estimator receives its third of the overall contract
    assert classifier.estimator_.time_limit_in_minutes == pytest.approx(2)
