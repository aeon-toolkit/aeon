"""Test check_random_state."""

import numpy as np
import pytest

from aeon.utils.validation import check_random_state


def test_none_does_not_return_global_generator():
    """None must not resolve to the process-global RandomState."""
    rng = check_random_state(None)

    assert isinstance(rng, np.random.RandomState)
    assert rng is not np.random.mtrand._rand


def test_global_generator_is_replaced():
    """Passing the global generator explicitly must also be replaced."""
    rng = check_random_state(np.random.mtrand._rand)

    assert rng is not np.random.mtrand._rand


def test_none_gives_independent_generators():
    """Two unseeded calls must not share state."""
    first = check_random_state(None)
    second = check_random_state(None)

    assert first is not second


@pytest.mark.parametrize("seed", [0, 42, 12345])
def test_int_seed_is_reproducible(seed):
    """An int seed must give the same stream every time."""
    first = check_random_state(seed).randint(0, 2**31 - 1, size=5)
    second = check_random_state(seed).randint(0, 2**31 - 1, size=5)

    np.testing.assert_array_equal(first, second)


def test_random_state_is_passed_through():
    """An explicit RandomState must be returned unchanged."""
    rng = np.random.RandomState(7)

    assert check_random_state(rng) is rng


def test_unseeded_is_not_deterministic():
    """Without a seed the generators must differ from one another."""
    first = check_random_state(None).randint(0, 2**31 - 1, size=10)
    second = check_random_state(None).randint(0, 2**31 - 1, size=10)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "estimator_name, attribute",
    [
        ("TimeSeriesKMeans", "_random_state"),
        ("TimeSeriesKMedoids", "_random_state"),
        ("KShape", "_rng"),
        ("TimeSeriesCLARA", "_random_state"),
        ("ElasticSOM", "_random_state"),
        ("KASBA", "_random_state"),
    ],
)
def test_clusterers_do_not_store_global_generator(estimator_name, attribute):
    """Unseeded clusterers must not keep the global RandomState as fitted state."""
    import aeon.clustering as clustering

    X = np.random.RandomState(0).random((14, 1, 24))

    estimator = getattr(clustering, estimator_name)(n_clusters=2)
    estimator.fit(X)

    assert getattr(estimator, attribute) is not np.random.mtrand._rand


@pytest.mark.parametrize(
    "estimator_name", ["SMOTE", "ADASYN", "RandomOverSampler", "ESMOTE"]
)
def test_samplers_do_not_store_global_generator(estimator_name):
    """Unseeded samplers must not keep the global RandomState as fitted state."""
    import aeon.transformations.collection.imbalance as imbalance

    X = np.random.RandomState(0).random((14, 1, 24))
    y = np.array([0] * 7 + [1] * 7)

    estimator = getattr(imbalance, estimator_name)()
    estimator.fit_transform(X, y)

    assert estimator._random_state is not np.random.mtrand._rand
