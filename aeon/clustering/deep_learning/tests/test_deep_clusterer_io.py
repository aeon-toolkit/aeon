"""Tests for saving and loading deep autoencoder clusterers."""

import numpy as np
import pytest
from pathlib import Path

from aeon.clustering.deep_learning._ae_dcnn import AEDCNNClusterer
from aeon.clustering.deep_learning._ae_drnn import AEDRNNClusterer
from aeon.utils.validation._dependencies import _check_soft_dependencies

from pathlib import Path

import numpy as np
import pytest

from aeon.clustering.deep_learning._ae_dcnn import AEDCNNClusterer
from aeon.clustering.deep_learning._ae_drnn import AEDRNNClusterer
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="TensorFlow soft dependency not found.",
)
@pytest.mark.parametrize("cls", [AEDRNNClusterer, AEDCNNClusterer])
def test_deep_clusterer_save_load(tmp_path, cls):
    """Test that deep autoencoder clusterers can load a saved Keras model."""
    X = np.random.randn(4, 10, 1).astype(np.float32)

    params = cls._get_test_params()[0]
    params["n_epochs"] = 1
    params["save_best_model"] = True
    params["file_path"] = str(tmp_path) + "/"

    estimator = params["estimator"]

    model = cls(**params)
    model.fit(X)

    saved_files = list(Path(tmp_path).glob("*.keras"))
    assert saved_files, "No .keras file was saved during model training"
    model_path = str(saved_files[0])

    loaded = cls(**params)
    loaded.load_model(model_path, estimator)

    assert loaded.model_ is not None
    assert hasattr(loaded.model_, "predict"), "Loaded keras model is not valid"
