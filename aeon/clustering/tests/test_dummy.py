"""Tests for DummyClusterer."""

import numpy as np
import pytest

from aeon.clustering import DummyClusterer


@pytest.mark.parametrize("strategy", ["random", "uniform", "single_cluster"])
def test_dummy_clusterer(strategy):
    """Test dummy clusterer basic functionalities."""
    model = DummyClusterer(strategy=strategy, n_clusters=3)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    model.fit(data)
    preds = model.predict(data)

    assert len(preds) == 3
    assert np.all(np.array([(pred < 3) for pred in preds]))
    assert np.all(np.array([(pred >= 0) for pred in preds]))


def test_dummy_clusterer_score():
    """Test score method of the dummy clusterer."""
    model = DummyClusterer(strategy="random")
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    model.fit(data)
    score = model.score(data)
    assert score is not None

    model = DummyClusterer(strategy="single_cluster")
    model.fit(data)
    score = model.score(data)
    assert score is not None
    assert score == 54.0
