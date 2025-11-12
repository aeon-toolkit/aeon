"""Test time series kGraph clustering."""

import numpy as np
import pytest
from aeon.clustering._k_graph import KGraphClusterer
from aeon.testing.data_generation import make_example_3d_numpy


def _run_kgraph_test(
    n_cases: int = 10,
    n_channels: int = 1,
    n_timepoints: int = 20,
    n_clusters: int = 2,
):
    """Utility function to train and validate a KGraphClusterer."""
    # generate simple equal-length data
    X = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=1,
        return_y=False,
    )

    model = KGraphClusterer(
        n_clusters=n_clusters,
        n_lengths=6,
        n_jobs=1,
        verbose=False,
    )

    labels = model.fit_predict(X)

    # --- general checks ---
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (n_cases,)
    assert np.unique(labels).size <= n_clusters
    assert hasattr(model, "graphs")
    assert isinstance(model.graphs, dict)
    assert hasattr(model, "labels_")
    assert np.array_equal(model.labels_, labels)

    # relevance arrays may exist
    assert model.all_lengths is not None
    assert isinstance(model.all_lengths, list)
    assert all(isinstance(l, (int, np.integer)) for l in model.all_lengths)

    # check optional methods
    interp = model.interprete(nb_patterns=1)
    assert isinstance(interp, dict)
    for k, v in interp.items():
        assert isinstance(k, (int, np.integer))
        assert isinstance(v, list)

    graphoids, names = model.compute_graphoids(mode="Proportion")
    assert isinstance(graphoids, np.ndarray)
    assert isinstance(names, list)
    assert graphoids.shape[0] == n_clusters

    return model


def test_kgraph_basic_runs():
    """Test that KGraphClusterer runs on basic synthetic data."""
    model = _run_kgraph_test(n_cases=6, n_channels=1, n_timepoints=15, n_clusters=2)
    assert isinstance(model.labels_, np.ndarray)
    assert model.labels_.shape == (6,)


def test_kgraph_predict_raises():
    """KGraphClusterer does not support out-of-sample predict."""
    X = make_example_3d_numpy(4, 1, 10, random_state=1, return_y=False)
    model = KGraphClusterer(n_clusters=2, n_lengths=3)
    model.fit(X)
    with pytest.raises(NotImplementedError):
        model.predict(X)


def test_kgraph_relevance_fields_exist():
    """After fitting, relevance attributes should be set or None."""
    X = make_example_3d_numpy(4, 1, 12, random_state=2, return_y=False)
    model = KGraphClusterer(n_clusters=2, n_lengths=4)
    model.fit(X)
    assert hasattr(model, "length_relevance")
    assert hasattr(model, "graph_relevance")
    assert hasattr(model, "relevance")
    assert hasattr(model, "optimal_length")