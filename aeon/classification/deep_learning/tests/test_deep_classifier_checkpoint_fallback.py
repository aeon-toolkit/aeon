"""Tests that deep classifiers fall back gracefully with no checkpoint file.

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

from aeon.classification.deep_learning._cnn import TimeCNNClassifier
from aeon.classification.deep_learning._disjoint_cnn import DisjointCNNClassifier
from aeon.classification.deep_learning._encoder import EncoderClassifier
from aeon.classification.deep_learning._fcn import FCNClassifier
from aeon.classification.deep_learning._inception_time import (
    IndividualInceptionClassifier,
)
from aeon.classification.deep_learning._lite_time import IndividualLITEClassifier
from aeon.classification.deep_learning._mlp import MLPClassifier
from aeon.classification.deep_learning._resnet import ResNetClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

ALL_DEEP_CLASSIFIERS = [
    IndividualInceptionClassifier,
    IndividualLITEClassifier,
    FCNClassifier,
    MLPClassifier,
    ResNetClassifier,
    EncoderClassifier,
    TimeCNNClassifier,
    DisjointCNNClassifier,
]


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("cls", ALL_DEEP_CLASSIFIERS)
def test_deep_classifier_missing_checkpoint_fallback(cls, monkeypatch):
    """Test fit completes when the best-model checkpoint is never written."""
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = os.path.join(temp, "")

        X, y = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=40)

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
