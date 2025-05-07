"""Tests for save/load functionality of InceptionTimeRegressor."""

import glob
import os
import tempfile

import numpy as np
import pytest

from aeon.regression.deep_learning import InceptionTimeRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_save_load_inceptiontime_regressor():
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
            n_epochs=1,
            random_state=42,
            save_best_model=True,
            file_path=temp_dir,
            n_regressors=1,
        )
        model.fit(X, y)

        y_pred_orig = model.predict(X)

        model_files = glob.glob(
            os.path.join(temp_dir, f"{model.best_file_name}*.keras")
        )

        loaded_model = InceptionTimeRegressor.load_model(model_paths=model_files)

        assert isinstance(loaded_model, InceptionTimeRegressor)

        preds = loaded_model.predict(X)
        assert isinstance(preds, np.ndarray)

        assert len(preds) == len(y)
        np.testing.assert_array_almost_equal(preds, y_pred_orig, decimal=4)
