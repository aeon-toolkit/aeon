"""Tests for save/load functionality of LiteTimeClassifier."""

import glob
import os
import tempfile

import numpy as np
import pytest

from aeon.classification.deep_learning import LITETimeClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_save_load_litetim():
    """Test saving and loading for LiteTimeClassifier."""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = os.path.join(temp, "")

        X, y = make_example_3d_numpy(
            n_cases=10, n_channels=1, n_timepoints=12, return_y=True
        )

        model = LITETimeClassifier(
            n_epochs=1, random_state=42, save_best_model=True, file_path=temp_dir
        )
        model.fit(X, y)

        y_pred_orig = model.predict(X)

        model_files = glob.glob(
            os.path.join(temp_dir, f"{model.best_file_name}*.keras")
        )

        loaded_model = LITETimeClassifier.load_model(
            model_path=model_files, classes=model.classes_
        )

        assert isinstance(loaded_model, LITETimeClassifier)

        preds = loaded_model.predict(X)
        assert isinstance(preds, np.ndarray)

        assert len(preds) == len(y)
        np.testing.assert_array_equal(preds, y_pred_orig)
