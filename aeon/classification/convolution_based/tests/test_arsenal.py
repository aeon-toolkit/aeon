"""Arsenal test code."""

import numpy as np
import pytest

from aeon.classification.convolution_based import Arsenal
from aeon.classification.convolution_based._arsenal import (
    _aggregate_class_votes,
    _get_oob_indices,
    _normalise_oob_probabilities,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)


def test_contracted_arsenal():
    """Contracted Arsenal builds at least two members and respects the cap."""
    X_train, y_train = make_example_3d_numpy()

    contract_max_n_estimators = 3
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=contract_max_n_estimators,
        n_kernels=20,
    )
    arsenal.fit(X_train, y_train)
    assert 1 < len(arsenal.estimators_) <= contract_max_n_estimators


@pytest.mark.parametrize("n_channels", [1, 4])
@pytest.mark.parametrize(
    ("rocket_transform", "expected_transformer"),
    [
        ("rocket", Rocket),
        ("minirocket", MiniRocket),
        ("multirocket", MultiRocket),
    ],
)
def test_arsenal_rocket_variants(rocket_transform, expected_transformer, n_channels):
    """Every ensemble member is built on the requested rocket transformer."""
    n_estimators = 2
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=n_channels, n_timepoints=50, random_state=0
    )

    clf = Arsenal(
        n_kernels=100,
        rocket_transform=rocket_transform,
        max_dilations_per_kernel=2,
        n_estimators=n_estimators,
        random_state=0,
    )
    clf.fit(X, y)

    assert len(clf.estimators_) == n_estimators
    for pipeline in clf.estimators_:
        assert isinstance(pipeline[0], expected_transformer)


def test_arsenal_invalid_rocket_transform():
    """An unknown rocket_transform raises at fit."""
    X, y = make_example_3d_numpy(n_cases=10, n_timepoints=20, random_state=0)

    with pytest.raises(ValueError, match="Invalid Rocket transformer"):
        Arsenal(rocket_transform="fubar").fit(X, y)


def test_arsenal_normalises_rocket_input_once(monkeypatch):
    """Input is normalised once per fit and once per predict, not per member."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=3, n_timepoints=30, random_state=0
    )
    original_fit_transform = Normalizer.fit_transform
    call_count = 0

    def counting_fit_transform(self, X, y=None):
        nonlocal call_count
        call_count += 1
        return original_fit_transform(self, X, y)

    monkeypatch.setattr(Normalizer, "fit_transform", counting_fit_transform)

    arsenal = Arsenal(
        n_kernels=20,
        n_estimators=3,
        random_state=0,
    ).fit(X, y)
    arsenal.predict_proba(X)

    assert call_count == 2


def test_arsenal_rocket_pipeline_accepts_raw_input():
    """Stored pipelines predict on raw input as Arsenal does on normalised input."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=3, n_timepoints=30, random_state=0
    )
    arsenal = Arsenal(
        n_kernels=20,
        n_estimators=2,
        random_state=0,
    ).fit(X, y)
    pipeline = arsenal.estimators_[0]
    rocket, scaler, ridge = (step[1] for step in pipeline.steps)

    expected = pipeline.predict(X)
    X_normalised = Normalizer().fit_transform(X)
    transformed = rocket._transform_kernels(X_normalised)
    actual = ridge.predict(scaler.transform(transformed))

    np.testing.assert_array_equal(actual, expected)


def test_arsenal_oob_indices_with_duplicates():
    """OOB indices are the sorted cases absent from a duplicated bootstrap."""
    subsample = np.array([4, 0, 4, 2, 0, 6, 2])

    oob = _get_oob_indices(subsample, n_cases=7)

    np.testing.assert_array_equal(oob, [1, 3, 5])


def test_arsenal_oob_probability_normalisation():
    """OOB rows divide by their available weight; unseen rows become uniform."""
    probabilities = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.25],
            [0.5, 0.25],
            [0.0, 0.0],
        ]
    )
    weights = [0.5, 0.25]
    oobs = [np.array([0, 2]), np.array([1, 2])]

    actual = _normalise_oob_probabilities(
        probabilities,
        weights,
        oobs,
        n_classes=2,
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2 / 3, 1 / 3],
            [0.5, 0.5],
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_arsenal_class_vote_aggregation():
    """Weighted class-index votes aggregate into the expected probabilities."""
    predictions = [
        np.array([0, 1, 1]),
        np.array([1, 1, 0]),
    ]
    weights = [0.75, 0.25]

    probabilities = _aggregate_class_votes(
        predictions,
        weights,
        n_cases=3,
        n_classes=2,
    )

    expected = np.array(
        [
            [0.75, 0.25],
            [0.0, 1.0],
            [0.25, 0.75],
        ]
    )
    np.testing.assert_array_equal(probabilities, expected)


def test_arsenal_fit_predict_returns_train_estimates():
    """Arsenal fit_predict labels come from valid OOB probability estimates."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=20, random_state=0
    )

    clf = Arsenal(n_kernels=5, n_estimators=2, random_state=0)
    proba = clf.fit_predict_proba(X, y)

    assert proba.shape == (len(y), clf.n_classes_)
    assert np.all(proba >= 0)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y)))

    predictions = Arsenal(n_kernels=5, n_estimators=2, random_state=0).fit_predict(X, y)
    assert predictions.shape == (len(y),)
    assert set(predictions).issubset(set(clf.classes_))


def test_arsenal_class_weight_reaches_ridge():
    """The class_weight parameter is passed to every member's ridge."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=30, random_state=0
    )

    clf = Arsenal(
        n_kernels=20, n_estimators=2, class_weight="balanced", random_state=0
    ).fit(X, y)

    for pipeline in clf.estimators_:
        assert pipeline[-1].class_weight == "balanced"


def test_arsenal_weights_are_cv_accuracies():
    """Ensemble weights are LOO CV accuracies, so they lie in [0, 1]."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=30, random_state=0
    )

    clf = Arsenal(n_kernels=20, n_estimators=3, random_state=0).fit(X, y)

    assert len(clf.weights_) == 3
    assert all(0 <= weight <= 1 for weight in clf.weights_)


def test_arsenal_n_jobs_does_not_change_output():
    """Threaded and sequential Arsenal fits produce identical results."""
    X, y = make_example_3d_numpy(
        n_cases=20, n_channels=1, n_timepoints=30, random_state=0
    )
    X_test, _ = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=30, random_state=1
    )

    sequential = Arsenal(n_kernels=20, n_estimators=3, random_state=0, n_jobs=1)
    threaded = Arsenal(n_kernels=20, n_estimators=3, random_state=0, n_jobs=2)

    sequential.fit(X, y)
    threaded.fit(X, y)
    assert threaded._n_jobs == 2

    np.testing.assert_array_equal(sequential.weights_, threaded.weights_)
    np.testing.assert_array_equal(
        sequential.predict_proba(X_test), threaded.predict_proba(X_test)
    )

    sequential_train = Arsenal(
        n_kernels=20, n_estimators=3, random_state=0, n_jobs=1
    ).fit_predict_proba(X, y)
    threaded_train = Arsenal(
        n_kernels=20, n_estimators=3, random_state=0, n_jobs=2
    ).fit_predict_proba(X, y)
    np.testing.assert_array_equal(sequential_train, threaded_train)
