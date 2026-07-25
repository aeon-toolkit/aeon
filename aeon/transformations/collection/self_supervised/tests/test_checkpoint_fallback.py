"""Tests that self-supervised transformers fall back gracefully.

With ``save_best_only=True`` the Keras ``ModelCheckpoint`` callback is not
guaranteed to have written a file by the time ``fit`` tries to load it back
(e.g. if the callback is skipped or training does not trigger a save). The
estimator should fall back to the in-memory trained model instead of
raising, regardless of the exact exception type an unwritten/missing file
triggers on a given TensorFlow/Keras backend.
"""

import glob
import os
import tempfile

import numpy as np
import pytest

from aeon.transformations.collection.self_supervised._time_mcl import TimeMCL
from aeon.transformations.collection.self_supervised._trilite import TRILITE
from aeon.utils.validation._dependencies import _check_soft_dependencies

ALL_SELF_SUPERVISED = [TRILITE, TimeMCL]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("cls", ALL_SELF_SUPERVISED)
def test_self_supervised_missing_checkpoint_fallback(cls, monkeypatch):
    """Test fit completes when the best-model checkpoint is never written."""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = os.path.join(temp, "")

        X = np.random.random((20, 2, 40)).astype(np.float32)

        # stop the ModelCheckpoint callback being added so no .keras file is
        # written during fit, simulating a checkpoint that was never saved
        monkeypatch.setattr(
            cls,
            "_get_model_checkpoint_callback",
            lambda self, callbacks, file_path, file_name: callbacks,
        )

        params = cls._get_test_params()
        if isinstance(params, list):
            params = params[0]
        params.update(
            {
                "n_epochs": 1,
                "random_state": 0,
                "file_path": temp_dir,
            }
        )

        model = cls(**params)
        model.fit(X=X)

        assert model.model_ is not None
        assert glob.glob(os.path.join(temp_dir, "*.keras")) == []

        X_transformed = model.transform(X=X)
        assert len(X_transformed) == len(X)
