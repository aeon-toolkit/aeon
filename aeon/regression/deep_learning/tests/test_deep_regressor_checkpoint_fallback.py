"""Tests that deep regressors fall back gracefully with no checkpoint file.

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

import pytest

from aeon.regression.deep_learning._cnn import TimeCNNRegressor
from aeon.regression.deep_learning._disjoint_cnn import DisjointCNNRegressor
from aeon.regression.deep_learning._encoder import EncoderRegressor
from aeon.regression.deep_learning._fcn import FCNRegressor
from aeon.regression.deep_learning._inception_time import (
    IndividualInceptionRegressor,
)
from aeon.regression.deep_learning._lite_time import IndividualLITERegressor
from aeon.regression.deep_learning._mlp import MLPRegressor
from aeon.regression.deep_learning._resnet import ResNetRegressor
from aeon.regression.deep_learning._rnn import RecurrentRegressor
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

ALL_DEEP_REGRESSORS = [
    IndividualInceptionRegressor,
    IndividualLITERegressor,
    FCNRegressor,
    MLPRegressor,
    ResNetRegressor,
    EncoderRegressor,
    TimeCNNRegressor,
    DisjointCNNRegressor,
    RecurrentRegressor,
]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("cls", ALL_DEEP_REGRESSORS)
def test_deep_regressor_missing_checkpoint_fallback(cls, monkeypatch):
    """Test fit completes when the best-model checkpoint is never written."""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = os.path.join(temp, "")

        X, y = make_example_3d_numpy(
            n_cases=10,
            n_channels=1,
            n_timepoints=40,
            return_y=True,
            regression_target=True,
        )

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
        model.fit(X, y)

        assert model.model_ is not None
        assert glob.glob(os.path.join(temp_dir, "*.keras")) == []

        preds = model.predict(X)
        assert len(preds) == len(y)
