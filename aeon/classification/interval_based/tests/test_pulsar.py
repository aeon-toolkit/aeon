"""Tests for PULSAR feature generation and classification."""

import numpy as np
import pytest

from aeon.classification.interval_based import PULSARClassifier
from aeon.classification.interval_based._pulsar import (
    _ar_order,
    _fisher_scores,
    _generate_interval_configurations,
    _local_statistic_matrix,
    _make_partitions,
    _pool_rows,
    _sliding_intervals,
)
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.utils.validation._dependencies import _check_soft_dependencies


def _small_classifier(**kwargs):
    params = {
        "representations": ("original",),
        "interval_lengths": (3,),
        "max_dilation": 2,
        "local_statistics": ("mean", "stdev", "slope", "min", "max"),
        "pooling_operators": ("max", "mean", "min"),
        "hierarchical_depth": 3,
        "n_random_pooling_operators": 2,
        "classifiers": ("ridge",),
        "random_state": 0,
    }
    params.update(kwargs)
    return PULSARClassifier(**params)


def test_pulsar_interval_configurations():
    """Only valid power-of-two dilations are generated in input order."""
    assert _generate_interval_configurations(20, (7, 9, 11), 16) == (
        (7, 1),
        (7, 2),
        (9, 1),
        (9, 2),
        (11, 1),
    )


def test_pulsar_dilated_interval_indexing():
    """Intervals use every valid start and the requested within-interval stride."""
    X = np.arange(10, dtype=float)[None, :]
    intervals = _sliding_intervals(X, length=3, dilation=2)
    np.testing.assert_array_equal(
        intervals[0],
        np.array(
            [
                [0, 2, 4],
                [1, 3, 5],
                [2, 4, 6],
                [3, 5, 7],
                [4, 6, 8],
                [5, 7, 9],
            ]
        ),
    )


def test_pulsar_local_statistics():
    """The seven local statistics have the published ordering and definitions."""
    X = np.array([[1, 2, 3, 4], [2, 2, 2, 2]], dtype=float)
    statistics = ("mean", "stdev", "slope", "min", "max", "iqr", "median")
    result = _local_statistic_matrix(X, statistics)

    np.testing.assert_allclose(result[0, :5], [2.5, np.sqrt(1.25), 1, 1, 4])
    np.testing.assert_allclose(result[1], [2, 0, 0, 2, 2, 0, 2])


def test_pulsar_autoregressive_order_rule():
    """The AR order uses the published truncated fourth-root rule."""
    X = np.zeros((2, 1, 100))
    assert _ar_order(X) == 12
    assert _ar_order(X[..., :20]) == int(12 * (20 / 100) ** 0.25)


def test_pulsar_pooling_operators():
    """Pooling operators handle regular and singleton response partitions."""
    X = np.array([[0, 1, 2, 3], [2, 2, 2, 2]], dtype=float)
    np.testing.assert_allclose(_pool_rows(X, "max"), [3, 2])
    np.testing.assert_allclose(_pool_rows(X, "mean"), [1.5, 2])
    np.testing.assert_allclose(_pool_rows(X, "min"), [0, 2])
    np.testing.assert_allclose(_pool_rows(X, "stdev"), [np.sqrt(1.25), 0])
    np.testing.assert_allclose(_pool_rows(X, "slope"), [1, 0])
    np.testing.assert_allclose(_pool_rows(X, "mean_crossing_count"), [1 / 3, 0])
    np.testing.assert_allclose(_pool_rows(X, "values_above_mean"), [0.5, 0])
    np.testing.assert_allclose(_pool_rows(X, "median"), [1.0078125, 2])
    np.testing.assert_allclose(_pool_rows(X, "iqr"), [1.96875, 0])
    np.testing.assert_allclose(_pool_rows(X[:1, :1], "mean_crossing_count"), [0])


def test_pulsar_hierarchical_partitions():
    """Partitions cover uneven response maps with half-open boundaries."""
    partitions = _make_partitions(5, 3)
    assert partitions[:3] == [(0, 5, 0), (0, 3, 1), (3, 5, 1)]
    assert partitions[3:] == [
        (0, 2, 2),
        (2, 3, 2),
        (3, 4, 2),
        (4, 5, 2),
    ]


def test_pulsar_random_pooling_selection_is_reproducible():
    """Fitted pooling choices are deterministic for a seed and differ across seeds."""
    first = _small_classifier(n_random_pooling_operators=1)
    first._validate_parameters()
    second = _small_classifier(n_random_pooling_operators=1)
    second._validate_parameters()
    first_state = first._new_interval_states(12, np.random.RandomState(0))
    second_state = second._new_interval_states(12, np.random.RandomState(0))
    other_state = second._new_interval_states(12, np.random.RandomState(1))
    assert first_state == second_state
    assert first_state != other_state


def test_pulsar_fisher_scores_and_tie_selection():
    """Fisher scores use training labels and stable candidate ordering."""
    X = np.array([[0, 1, 10], [0, 1, 10], [1, 1, 10], [1, 1, 10]], dtype=float)
    y = np.array(["left", "left", "right", "right"])
    scores = _fisher_scores(X, y)
    assert scores[0] > scores[1]
    assert scores[2] == 0

    classifier = _small_classifier(feature_selection_percentage=1)
    X_train = np.arange(20, dtype=float).reshape(4, 1, 5)
    classifier.fit(X_train, np.array([0, 0, 1, 1]))
    assert classifier.selected_feature_indices_.shape == (1,)


def test_pulsar_feature_order_is_consistent_between_fit_and_prediction():
    """Prediction regenerates exactly the fitted global/candidate column layout."""
    X = np.arange(60, dtype=float).reshape(12, 1, 5)
    y = np.array([0, 1] * 6)
    classifier = _small_classifier().fit(X, y)
    _, candidates, _ = classifier._feature_transform(X[:2], fitting=False)
    assert candidates.shape[1] == classifier.n_candidate_features_
    assert classifier.global_feature_metadata_[0].representation == "original"
    assert classifier.candidate_feature_metadata_[-1].local_statistic == "raw"
    np.testing.assert_allclose(classifier._fit_candidate_features_[:, -5:], X[:, 0, :])


def test_pulsar_short_series_uses_available_representations():
    """Very short series remain usable when an AR representation is impossible."""
    X = np.arange(40, dtype=float).reshape(8, 1, 5)
    y = np.array([0, 1] * 4)
    classifier = PULSARClassifier(
        representations=("original", "autoregressive"),
        interval_lengths=(3,),
        max_dilation=1,
        classifiers=("ridge",),
        random_state=0,
    ).fit(X, y)
    assert [state.name for state in classifier._representation_states_] == ["original"]
    assert np.isfinite(classifier.predict_proba(X[:2])).all()


@pytest.mark.parametrize(
    "labels",
    [np.array([3, 3, 7, 7] * 2), np.array(["a", "a", "z", "z"] * 2)],
)
def test_pulsar_binary_label_types(labels):
    """Non-zero-based integers and strings are returned in class order."""
    X = np.arange(len(labels) * 5, dtype=float).reshape(len(labels), 1, 5)
    classifier = _small_classifier().fit(X, labels)
    probabilities = classifier.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert set(classifier.predict(X[:3])).issubset(set(labels))


def test_pulsar_multiclass_probabilities():
    """The averaged calibrated heads support multiclass labels."""
    X = np.arange(90, dtype=float).reshape(18, 1, 5)
    y = np.array(["a", "b", "c"] * 6)
    classifier = _small_classifier().fit(X, y)
    probabilities = classifier.predict_proba(X[:4])
    assert probabilities.shape == (4, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_pulsar_no_test_leakage_in_feature_selection():
    """Changing test values cannot alter fitted selection metadata or scores."""
    X = np.arange(60, dtype=float).reshape(12, 1, 5)
    y = np.array([0, 1] * 6)
    classifier = _small_classifier().fit(X, y)
    selected = classifier.selected_feature_indices_.copy()
    scores = classifier.feature_selection_scores_.copy()
    classifier.predict(X[:2] + 10000)
    np.testing.assert_array_equal(classifier.selected_feature_indices_, selected)
    np.testing.assert_array_equal(classifier.feature_selection_scores_, scores)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip the four-representation test without statsmodels",
)
def test_pulsar_default_representations():
    """The default fit uses all four representations when AR support is available."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]
    classifier = PULSARClassifier(
        interval_lengths=(3,),
        max_dilation=1,
        n_estimators=2,
        random_state=0,
    ).fit(X, y)
    assert [state.name for state in classifier._representation_states_] == [
        "original",
        "periodogram",
        "derivative",
        "autoregressive",
    ]
