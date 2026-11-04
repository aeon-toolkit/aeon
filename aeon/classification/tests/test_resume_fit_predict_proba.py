"""Tests for resume_fit_predict_proba, train estimates continued from a checkpoint."""

import numpy as np
import pytest

from aeon.classification.convolution_based import Arsenal
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.testing.data_generation import make_example_3d_numpy

# estimator, constructor parameters, and the size limit grown on resume
GROWABLE = [
    (Arsenal, dict(n_kernels=10, random_state=0), "n_estimators", 2, 4),
    (DrCIFClassifier, dict(random_state=0), "n_estimators", 2, 4),
    (
        TemporalDictionaryEnsemble,
        dict(max_ensemble_size=3, random_state=0),
        "n_parameter_samples",
        4,
        8,
    ),
]


@pytest.fixture
def _data():
    """Provide a small deterministic classification dataset."""
    return make_example_3d_numpy(n_cases=12, n_timepoints=24, random_state=42)


@pytest.mark.parametrize("cls, params, limit, initial, target", GROWABLE)
def test_resume_fit_predict_proba_matches_full_fit(
    _data, checkpoint_directory, cls, params, limit, initial, target
):
    """Train estimates from a resumed fit equal those of an uninterrupted fit."""
    X, y = _data
    full = cls(**params, **{limit: target})
    expected = full.fit_predict_proba(X, y)

    partial = cls(**params, **{limit: initial})
    partial.fit_predict_proba(X, y)
    path = checkpoint_directory / f"{cls.__name__}.pkl"
    partial.save_checkpoint(path)
    restored = cls.load_checkpoint(path)
    restored.set_params(**{limit: target})

    train = restored.resume_fit_predict_proba(X, y)
    assert restored.is_fitted
    np.testing.assert_allclose(train, expected)
    np.testing.assert_allclose(restored.predict_proba(X), full.predict_proba(X))


def test_rotation_forest_resume_fit_predict_proba(checkpoint_directory):
    """A rotation forest grown on resume gives the OOB estimates of a full fit."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=12, random_state=0
    )
    X = X[:, 0]
    full = RotationForestClassifier(n_estimators=6, random_state=0)
    expected = full.fit_predict_proba(X, y)
    partial = RotationForestClassifier(n_estimators=3, random_state=0)
    partial.fit_predict_proba(X, y)
    path = checkpoint_directory / "rotf.pkl"
    partial.save_checkpoint(path)
    restored = RotationForestClassifier.load_checkpoint(path)
    restored.set_params(n_estimators=6)
    np.testing.assert_allclose(restored.resume_fit_predict_proba(X, y), expected)


@pytest.mark.parametrize(
    "estimator",
    [
        Arsenal(n_kernels=10, n_estimators=2, random_state=0),
        DrCIFClassifier(n_estimators=2, random_state=0),
        TemporalDictionaryEnsemble(n_parameter_samples=4, random_state=0),
        ShapeletTransformClassifier(
            n_shapelet_samples=50,
            estimator=RotationForestClassifier(n_estimators=3, random_state=0),
            random_state=0,
        ),
    ],
)
def test_resume_fit_predict_proba_needs_train_estimate_fit(_data, estimator):
    """A fit begun with fit has no train estimates to continue."""
    X, y = _data
    estimator.fit(X, y)
    with pytest.raises(ValueError, match="no train estimates to continue"):
        estimator.resume_fit_predict_proba(X, y)


def test_resume_fit_predict_proba_needs_checkpointing(_data):
    """Classifiers without checkpointing do not offer it."""
    from aeon.testing.mock_estimators import MockClassifier

    X, y = _data
    clf = MockClassifier().fit(X, y)
    with pytest.raises(NotImplementedError):
        clf.resume_fit_predict_proba(X, y)
