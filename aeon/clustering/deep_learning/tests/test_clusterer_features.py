"""Tests whether various clusterer params work well."""

import numpy as np
import pytest

from aeon.clustering.deep_learning import AEFCNClusterer, AEResNetClusterer
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.clustering.dummy import DummyClusterer


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_multi_rec_fcn():
    """Tests whether multi-rec loss works fine or not."""
    X = np.random.random((100, 5, 2))
    clst = AEFCNClusterer(
        n_epochs = 5,
        batch_size = 4,
        use_bias = False,
        n_layers= 2,
        n_filters = [2,2],
        kernel_size = [2,2],
        padding = "same",
        strides = 1,
        latent_space_dim = 4,
        estimator= DummyClusterer(n_clusters=2),
        loss="multi_rec",
    )
    clst.fit(X)
    assert (
        clst.history["loss"][0] > clst.history["loss"][4]
    )  # Check if loss is decreasing.
    clst = AEResNetClusterer(
        n_epochs = 5,
        batch_size = 4,
        n_residual_blocks = 2,
        n_conv_per_residual_block = 1,
        n_filters = [2,2],
        kernel_size = 2,
        use_bias = False,
        estimator = DummyClusterer(n_clusters=2),
        loss="multi_rec",
    )
    clst.fit(X)
    assert clst.history["loss"][0] > clst.history["loss"][4]
