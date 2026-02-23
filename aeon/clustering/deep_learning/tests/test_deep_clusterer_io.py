"""Tests for loading Keras models in deep autoencoder clusterers."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon.clustering.deep_learning._ae_abgru import AEAttentionBiGRUClusterer
from aeon.clustering.deep_learning._ae_bgru import AEBiGRUClusterer
from aeon.clustering.deep_learning._ae_dcnn import AEDCNNClusterer
from aeon.clustering.deep_learning._ae_drnn import AEDRNNClusterer
from aeon.clustering.deep_learning._ae_fcn import AEFCNClusterer
from aeon.clustering.deep_learning._ae_resnet import AEResNetClusterer
from aeon.utils.validation._dependencies import _check_soft_dependencies

ALL_DEEP_CLUSTERERS = [
    AEAttentionBiGRUClusterer,
    AEBiGRUClusterer,
    AEDCNNClusterer,
    AEDRNNClusterer,
    AEFCNClusterer,
    AEResNetClusterer,
]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="TensorFlow not installed.",
)
@pytest.mark.parametrize("cls", ALL_DEEP_CLUSTERERS)
def test_deep_clusterer_load_model(cls):
    """Test that all deep autoencoder clusterers load saved Keras models correctly."""
    X = np.random.randn(4, 10, 1).astype(np.float32)
    params = cls._get_test_params()[0]
    params["n_epochs"] = 1
    params["save_best_model"] = True

    with tempfile.TemporaryDirectory() as tmp:
        params["file_path"] = tmp + "/"
        model = cls(**params)
        model.fit(X)
        trained_estimator = model._estimator
        saved = list(Path(tmp).glob("*.keras"))
        assert saved, f"No .keras file saved for {cls.__name__}"
        model_path = str(saved[0])
        loaded = cls(**params)
        loaded.load_model(model_path, trained_estimator)
        assert loaded.model_ is not None, f"Loaded model_ is None for {cls.__name__}"
        assert hasattr(loaded.model_, "predict"), f"Invalid model_ for {cls.__name__}"
        preds = loaded.predict(X)
        assert preds.shape[0] == X.shape[0], f"Predict failed for {cls.__name__}"
