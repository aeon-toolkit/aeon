"""Tests whether various clusterer params work well."""

import numpy as np

from aeon.clustering.deep_learning import AEFCNClusterer


def test_multi_rec_fcn():
    """Tests whether multi-rec loss works fine or not."""
    X = np.random.random((100, 5, 2))
    clst = AEFCNClusterer(n_clusters=2, n_epochs=10, loss="multi_rec")
    clst.fit(X)
    assert clst.history["loss"][0] > clst.history["loss"][1]  # Check if loss is decreasing.
