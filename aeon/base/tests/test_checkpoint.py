"""Tests for atomic checkpoint persistence and scheduling."""

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from aeon.base import CheckpointableMixin
from aeon.testing.mock_estimators import MockClassifier


@pytest.fixture
def checkpoint_directory():
    """Create and clean up an isolated directory for checkpoint files."""
    with TemporaryDirectory() as directory:
        yield Path(directory)


class _Checkpointable(CheckpointableMixin, MockClassifier):
    """Small estimator with explicit safe continuation state."""


def _ready_estimator():
    estimator = _Checkpointable()
    estimator._checkpoint_ready = True
    estimator.completed_ = [1, 2, 3]
    return estimator


def test_checkpoint_round_trip(checkpoint_directory):
    """Restoring a checkpoint creates an independent object with all its state."""
    estimator = _ready_estimator()
    path = checkpoint_directory / "checkpoint.pkl"
    estimator.save_checkpoint(str(path))
    restored = _Checkpointable.load_checkpoint(path)
    assert restored.completed_ == [1, 2, 3]
    assert restored is not estimator


@pytest.mark.parametrize("failure", ["serialization", "replace"])
def test_failed_save_preserves_checkpoint(checkpoint_directory, failure):
    """A failed save preserves the previous file and removes temporary files."""
    estimator = _ready_estimator()
    path = checkpoint_directory / "checkpoint.pkl"
    estimator.save_checkpoint(path)
    original = path.read_bytes()
    if failure == "serialization":
        estimator.unpickleable_ = lambda: None
        with pytest.raises((AttributeError, pickle.PicklingError)):
            estimator.save_checkpoint(path)
    else:
        with patch("aeon.base._checkpoint.os.replace", side_effect=OSError("disk")):
            with pytest.raises(OSError, match="disk"):
                estimator.save_checkpoint(path)
    assert path.read_bytes() == original
    assert _Checkpointable.load_checkpoint(path).completed_ == [1, 2, 3]
    assert list(checkpoint_directory.iterdir()) == [path]


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("format_version", 999, "format version"),
        ("aeon_version", "0.0.0", "aeon version"),
        ("estimator_class", "wrong.Class", "estimator class"),
    ],
)
def test_checkpoint_metadata(checkpoint_directory, field, value, message):
    """Reject incompatible checkpoint metadata."""
    path = checkpoint_directory / "checkpoint.pkl"
    _ready_estimator().save_checkpoint(path)
    with path.open("rb") as file:
        metadata = pickle.load(file)
        estimator = pickle.load(file)
    metadata[field] = value
    with path.open("wb") as file:
        pickle.dump(metadata, file)
        pickle.dump(estimator, file)
    with pytest.raises(ValueError, match=message):
        _Checkpointable.load_checkpoint(path)


def test_checkpoint_rejects_wrong_class(checkpoint_directory):
    """Loading via an unrelated estimator type fails clearly."""
    from aeon.classification.convolution_based import Arsenal

    path = checkpoint_directory / "checkpoint.pkl"
    _ready_estimator().save_checkpoint(path)
    with pytest.raises(TypeError, match="expected an instance"):
        Arsenal.load_checkpoint(path)


def test_checkpoint_requires_safe_pickleable_state(checkpoint_directory):
    """Respect cant_pickle and reject absent or in-flight state."""
    estimator = _ready_estimator()
    estimator.set_tags(cant_pickle=True)
    with pytest.raises(ValueError, match="cant_pickle"):
        estimator.save_checkpoint(checkpoint_directory / "checkpoint.pkl")
    estimator.set_tags(cant_pickle=False)
    estimator._checkpoint_ready = False
    with pytest.raises(ValueError, match="safe continuation"):
        estimator.save_checkpoint(checkpoint_directory / "checkpoint.pkl")


def test_checkpoint_timer(checkpoint_directory):
    """Intervals are in minutes and restarted for each invocation."""
    estimator = _ready_estimator()
    estimator.checkpoint_path = checkpoint_directory / "checkpoint.pkl"
    estimator.checkpoint_interval = 2
    with patch("aeon.base._checkpoint.time.monotonic", return_value=10):
        estimator._start_checkpoint_timer()
    with patch.object(estimator, "save_checkpoint") as save:
        with patch("aeon.base._checkpoint.time.monotonic", return_value=129):
            estimator._checkpoint_if_due()
            save.assert_not_called()
        with patch("aeon.base._checkpoint.time.monotonic", return_value=130):
            estimator._checkpoint_if_due()
            save.assert_called_once_with(estimator.checkpoint_path)
            estimator._start_checkpoint_timer()
        save.reset_mock()
        with patch("aeon.base._checkpoint.time.monotonic", return_value=131):
            estimator._checkpoint_if_due()
            save.assert_not_called()
            estimator._checkpoint_if_due(force=True)
            save.assert_called_once()


@pytest.mark.parametrize("interval", [-1, 0, float("inf"), float("nan"), True, "1"])
def test_invalid_checkpoint_interval(interval):
    """Invalid checkpoint intervals fail before training begins."""
    estimator = _ready_estimator()
    estimator.checkpoint_path = None
    estimator.checkpoint_interval = interval
    with pytest.raises(ValueError, match="checkpoint_interval"):
        estimator._start_checkpoint_timer()
