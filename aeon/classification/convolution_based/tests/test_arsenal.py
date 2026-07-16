"""Arsenal test code."""

import numpy as np
import pytest

from aeon.classification.convolution_based import Arsenal
from aeon.classification.convolution_based._arsenal import (
    _aggregate_class_votes,
    _get_oob_indices,
    _normalise_oob_probabilities,
)
from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)


def test_contracted_arsenal():
    """Test of contracted Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = make_example_3d_numpy()
    # train contracted Arsenal
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=3,
        n_kernels=20,
    )
    arsenal.fit(X_train, y_train)
    assert len(arsenal.estimators_) > 1


def test_arsenal():
    """Test correct rocket variant is selected."""
    X_train, y_train = make_example_2d_numpy_collection(n_cases=20, n_timepoints=50)
    afc = Arsenal(n_kernels=20, n_estimators=2)
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], Rocket)
    assert len(afc.estimators_) == 2
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="minirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MiniRocket)
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="multirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MultiRocket)
    X_train, y_train = make_example_3d_numpy(n_cases=20, n_timepoints=50, n_channels=4)
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="minirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MiniRocket)
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="multirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MultiRocket)
    afc = Arsenal(rocket_transform="fubar")
    with pytest.raises(ValueError, match="Invalid Rocket transformer: fubar"):
        afc.fit(X_train, y_train)


def test_arsenal_normalises_rocket_input_once(monkeypatch):
    """Arsenal should share normalization across its ROCKET ensemble members."""
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
    """Stored ROCKET pipelines should retain their raw-input prediction behavior."""
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


def test_arsenal_vectorised_oob_probability_normalisation():
    """OOB rows should use their available weight or uniform probabilities."""
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


def test_arsenal_compact_class_vote_aggregation():
    """Compact class-index outputs should aggregate without dense worker matrices."""
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
