"""Tests whether various clusterer params work well."""

import numpy as np

from aeon.clustering.deep_learning import AEFCNClusterer


def test_multi_rec_fcn():
    """Tests whether multi-rec loss works fine or not."""
    X = np.random.random((100, 5, 2))
    clst = AEFCNClusterer(n_clusters=2, n_epochs=10, loss="multi_rec")
    _history = clst.fit(X)
    assert _history["loss"][0] > _history["loss"][1]  # Check if loss is decreasing.
