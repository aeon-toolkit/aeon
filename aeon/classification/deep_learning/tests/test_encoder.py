"""Tests for EncoderClassifier."""

import os
import sys
from types import SimpleNamespace

import numpy as np

from aeon.classification.deep_learning import EncoderClassifier


def test_encoder_uses_training_model_when_checkpoint_missing(monkeypatch, tmp_path):
    """Test EncoderClassifier handles a missing best-checkpoint file."""

    class _ModelCheckpoint:
        def __init__(self, filepath, monitor, save_best_only):
            self.filepath = filepath
            self.monitor = monitor
            self.save_best_only = save_best_only

    class _Models:
        load_model_called = False

        @classmethod
        def load_model(cls, model_path, compile=False):
            cls.load_model_called = True
            raise AssertionError(f"load_model should not be called for {model_path}")

    fake_tensorflow = SimpleNamespace(
        keras=SimpleNamespace(
            callbacks=SimpleNamespace(ModelCheckpoint=_ModelCheckpoint),
            models=_Models,
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)

    fallback_model = object()

    class _TrainingModel:
        def fit(self, X, y, batch_size, epochs, verbose, callbacks):
            return SimpleNamespace(history={"loss": [0.0]})

        def __deepcopy__(self, memo):
            return fallback_model

    classifier = EncoderClassifier(
        batch_size=2,
        file_path=str(tmp_path) + os.sep,
        n_epochs=1,
        random_state=0,
    )
    monkeypatch.setattr(
        classifier,
        "build_model",
        lambda input_shape, n_classes: _TrainingModel(),
    )

    X = np.zeros((4, 1, 5))
    y = np.array([0, 1, 0, 1])

    fitted = classifier._fit(X, y)

    assert fitted is classifier
    assert classifier.model_ is fallback_model
    assert not _Models.load_model_called
