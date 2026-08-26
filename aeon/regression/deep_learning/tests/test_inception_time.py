"""Tests for save/load functionality of InceptionTimeRegressor."""

import glob
import os
import tempfile

import numpy as np
import pytest

from aeon.regression.deep_learning import (
    InceptionTimeRegressor,
    IndividualInceptionRegressor,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_save_load_inceptiontime():
    """Test saving and loading for InceptionTimeRegressor."""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = os.path.join(temp, "")

        X, y = make_example_3d_numpy(
            n_cases=10,
            n_channels=1,
            n_timepoints=12,
            return_y=True,
            regression_target=True,
        )

        model = InceptionTimeRegressor(
            n_epochs=1, random_state=42, save_best_model=True, file_path=temp_dir
        )
        model.fit(X, y)

        y_pred_orig = model.predict(X)

        model_file = glob.glob(os.path.join(temp_dir, f"{model.best_file_name}*.keras"))

        loaded_model = InceptionTimeRegressor.load_model(model_path=model_file)

        assert isinstance(loaded_model, InceptionTimeRegressor)

        preds = loaded_model.predict(X)
        assert isinstance(preds, np.ndarray)

        assert len(preds) == len(y)
        np.testing.assert_array_equal(preds, y_pred_orig)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_inception_missing_checkpoint_fallback(monkeypatch):
    """Test fit completes when the best-model checkpoint is never written."""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = os.path.join(temp, "")

        X, y = make_example_3d_numpy(
            n_cases=10,
            n_channels=1,
            n_timepoints=12,
            return_y=True,
            regression_target=True,
        )

        # stop the ModelCheckpoint callback being added so no .keras file is
        # written during fit, simulating a checkpoint that was never saved
        monkeypatch.setattr(
            IndividualInceptionRegressor,
            "_get_model_checkpoint_callback",
            lambda self, callbacks, file_path, file_name: callbacks,
        )

        model = IndividualInceptionRegressor(
            n_epochs=1,
            batch_size=4,
            depth=1,
            kernel_size=4,
            use_residual=False,
            random_state=42,
            file_path=temp_dir,
            callbacks=[],
        )
        model.fit(X, y)

        assert model.model_ is not None
        assert glob.glob(os.path.join(temp_dir, "*.keras")) == []

        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
