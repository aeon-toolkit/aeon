"""Tests whether various clusterer params work well."""

import numpy as np
import pytest

from aeon.clustering.deep_learning import (
    AEAttentionBiGRUClusterer,
    AEBiGRUClusterer,
    AEDCNNClusterer,
    AEDRNNClusterer,
    AEFCNClusterer,
    AEResNetClusterer,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_multi_rec_fcn():
    """Tests whether multi-rec loss works fine or not."""
    X = np.random.random((100, 5, 2))
    clst = AEFCNClusterer(**AEFCNClusterer._get_test_params()[0], loss="multi_rec")
    clst.fit(X)
    assert isinstance(clst.history["loss"][-1], float)

    clst = AEResNetClusterer(
        **AEResNetClusterer._get_test_params()[0], loss="multi_rec"
    )
    clst.fit(X)
    assert isinstance(clst.history["loss"][-1], float)


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency not found.",
)
def test_validation_split():
    """Test validation_split parameter."""
    X = np.random.random((100, 5, 2))

    models = [
        AEFCNClusterer,
        AEResNetClusterer,
        AEDCNNClusterer,
        AEDRNNClusterer,
        AEAttentionBiGRUClusterer,
        AEBiGRUClusterer,
    ]

    for model in models:
        # Test with validation_split = 0 (no validation)
        clst = model(
            **model._get_test_params()[0],
            validation_split=0,
        )
        clst.fit(X)
        assert "val_loss" not in clst.history.history or all(
            v is None for v in clst.history.history["val_loss"]
        )

        # Test with validation_split = 0.2 (20% validation)
        clst = model(
            **model._get_test_params()[0],
            validation_split=0.2,
        )
        clst.fit(X)
        assert "val_loss" in clst.history.history
        assert len(clst.history.history["val_loss"]) == len(
            clst.history.history["loss"]
        )
        assert all(isinstance(v, float) for v in clst.history.history["val_loss"])
