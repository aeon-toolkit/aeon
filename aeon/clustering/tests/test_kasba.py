"""Test KASBA."""

import numpy as np
import pytest

from aeon.clustering import KASBA
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize("n_channels", [1, 3])
def test_kasba_fit_structure(n_channels):
    """Fitted KASBA produces a valid partition, centers and diagnostics."""
    n_cases, n_timepoints, n_clusters = 20, 10, 2
    X = make_example_3d_numpy(
        n_cases, n_channels, n_timepoints, random_state=1, return_y=False
    )

    kasba = KASBA(n_clusters=n_clusters, random_state=1)
    kasba.fit(X)

    # labels form a valid partition over the requested clusters
    assert kasba.labels_.shape == (n_cases,)
    assert np.issubdtype(kasba.labels_.dtype, np.integer)
    assert set(np.unique(kasba.labels_)) <= set(range(n_clusters))
    # with more cases than clusters no cluster should be empty
    assert len(np.unique(kasba.labels_)) == n_clusters

    # centers have one average series per cluster
    assert kasba.cluster_centers_.shape == (n_clusters, n_channels, n_timepoints)
    assert np.all(np.isfinite(kasba.cluster_centers_))

    # fit diagnostics
    assert np.isfinite(kasba.inertia_) and kasba.inertia_ >= 0
    assert kasba.n_iter_ >= 1


@pytest.mark.parametrize("n_channels", [1, 3])
def test_kasba_predict_consistent_with_fit(n_channels):
    """Predict assigns training data to the same clusters as fit."""
    X = make_example_3d_numpy(20, n_channels, 10, random_state=1, return_y=False)

    kasba = KASBA(n_clusters=2, random_state=1)
    kasba.fit(X)

    preds = kasba.predict(X)
    assert preds.shape == (20,)
    # on convergence, assignment to the final centers reproduces labels_
    assert np.array_equal(preds, kasba.labels_)


def test_kasba_reproducible_with_random_state():
    """The same random_state yields identical labels and centers."""
    X = make_example_3d_numpy(20, 1, 10, random_state=1, return_y=False)

    first = KASBA(n_clusters=2, random_state=7).fit(X)
    second = KASBA(n_clusters=2, random_state=7).fit(X)

    assert np.array_equal(first.labels_, second.labels_)
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
