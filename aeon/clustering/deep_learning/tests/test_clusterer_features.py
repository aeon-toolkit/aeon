"""Tests whether various clusterer params work well."""

import numpy as np
import pytest

from aeon.clustering.deep_learning import AEFCNClusterer, AEResNetClusterer
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_multi_rec_fcn():
    """Tests whether multi-rec loss works fine or not."""
    X = np.random.random((100, 5, 2))
    clst = AEFCNClusterer(
        n_clusters=2, n_epochs=10, n_filters=[2, 3, 4], loss="multi_rec"
    )
    clst.fit(X)
    assert (
        clst.history["loss"][0] > clst.history["loss"][9]
    )  # Check if loss is decreasing.
    clst = AEResNetClusterer(n_clusters=2, n_epochs=10, loss="multi_rec")
    clst.fit(X)
    assert clst.history["loss"][0] > clst.history["loss"][9]
