"""Test load_model functionality for deep learning regression ensemble models."""
import os
import tempfile

import numpy as np
import pytest

from aeon.regression.deep_learning import InceptionTimeRegressor, LITETimeRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_inception_time_regressor_load_model():
    """Test loading InceptionTimeRegressor models from files."""
    with tempfile.TemporaryDirectory() as tmp:
        if tmp[-1] != "/":
            tmp = tmp + "/"

        # Generate sample data
        X_train, y_train = make_example_3d_numpy(
            n_cases=10, n_channels=1, n_timepoints=50, regression_target=True
        )
        X_test = X_train.copy()

        # Train model with both best and last model saving
        reg = InceptionTimeRegressor(
            n_epochs=2,
            batch_size=4,
            n_regressors=2,
            save_best_model=True,
            save_last_model=True,
            file_path=tmp,
        )
        reg.fit(X_train, y_train)

        # Get predictions from original model
        y_pred_orig = reg.predict(X_test)

        # Load and test best model
        reg_best = InceptionTimeRegressor.load_model(tmp, load_best=True)
        y_pred_best = reg_best.predict(X_test)
        np.testing.assert_array_almost_equal(y_pred_orig, y_pred_best)

        # Load and test last model
        reg_last = InceptionTimeRegressor.load_model(tmp, load_best=False)
        y_pred_last = reg_last.predict(X_test)
        assert len(reg_last.regressors_) == reg.n_regressors

        # Test error case with invalid path
        with pytest.raises(FileNotFoundError):
            InceptionTimeRegressor.load_model("invalid/path/")


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_lite_time_regressor_load_model():
    """Test loading LITETimeRegressor models from files."""
    with tempfile.TemporaryDirectory() as tmp:
        if tmp[-1] != "/":
            tmp = tmp + "/"

        # Generate sample data
        X_train, y_train = make_example_3d_numpy(
            n_cases=10, n_channels=1, n_timepoints=50, regression_target=True
        )
        X_test = X_train.copy()

        # Train model with both best and last model saving
        reg = LITETimeRegressor(
            n_epochs=2,
            batch_size=4,
            n_regressors=2,
            save_best_model=True,
            save_last_model=True,
            file_path=tmp,
        )
        reg.fit(X_train, y_train)

        # Get predictions from original model
        y_pred_orig = reg.predict(X_test)

        # Load and test best model
        reg_best = LITETimeRegressor.load_model(tmp, load_best=True)
        y_pred_best = reg_best.predict(X_test)
        np.testing.assert_array_almost_equal(y_pred_orig, y_pred_best)

        # Load and test last model
        reg_last = LITETimeRegressor.load_model(tmp, load_best=False)
        y_pred_last = reg_last.predict(X_test)
        assert len(reg_last.regressors_) == reg.n_regressors

        # Test error case with invalid path
        with pytest.raises(FileNotFoundError):
            LITETimeRegressor.load_model("invalid/path/")