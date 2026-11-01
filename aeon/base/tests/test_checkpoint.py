"""Tests for atomic checkpoint persistence and scheduling."""

import json
import pickle
from copy import deepcopy
from unittest.mock import patch

import pytest

from aeon import __version__
from aeon.base import CheckpointableMixin
from aeon.testing.mock_estimators import MockClassifier


class _Checkpointable(CheckpointableMixin, MockClassifier):
    """Small estimator with explicit safe continuation state."""


def _ready_estimator():
    estimator = _Checkpointable()
    estimator._checkpoint_ready = True
    estimator.completed_ = [1, 2, 3]
    return estimator


def _update_checkpoint_metadata(path, **changes):
    with path.open("rb") as file:
        metadata = json.loads(file.readline())
        body = file.read()
    metadata.update(changes)
    with path.open("wb") as file:
        file.write(json.dumps(metadata).encode("utf-8") + b"\n")
        file.write(body)


def test_checkpoint_round_trip(checkpoint_directory):
    """Copying preserves paths; restored continuation state is independent."""
    estimator = _ready_estimator()
    path = checkpoint_directory / "checkpoint.pkl"
    estimator.checkpoint_path = path
    assert deepcopy(estimator).checkpoint_path == path
    estimator.save_checkpoint(str(path))
    restored = _Checkpointable.load_checkpoint(path)
    assert restored.checkpoint_path == path
    assert restored.completed_ == estimator.completed_
    restored.completed_.append(4)
    assert estimator.completed_ == [1, 2, 3]


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
        ("estimator_class", "wrong.Class", "estimator class"),
        ("parameter_signature", "not-the-fitted-signature", "parameter signature"),
    ],
)
def test_checkpoint_metadata(checkpoint_directory, field, value, message):
    """Reject incompatible checkpoint metadata."""
    path = checkpoint_directory / "checkpoint.pkl"
    _ready_estimator().save_checkpoint(path)
    _update_checkpoint_metadata(path, **{field: value})
    with pytest.raises(ValueError, match=message):
        _Checkpointable.load_checkpoint(path)


def test_read_checkpoint_metadata(checkpoint_directory):
    """The public reader returns the raw JSON header without unpickling."""
    path = checkpoint_directory / "checkpoint.pkl"
    estimator = _ready_estimator()
    estimator._checkpoint_parameter_signature = "a-parameter-hash"
    estimator.save_checkpoint(path)

    with path.open("rb") as file:
        metadata = json.loads(file.readline())

    with patch("aeon.base._checkpoint.pickle.load") as load:
        assert CheckpointableMixin.read_checkpoint_metadata(path) == metadata
        load.assert_not_called()

    assert metadata["format_version"] == 1
    assert metadata["aeon_version"] == __version__
    assert metadata["estimator_class"].endswith("_Checkpointable")
    assert metadata["parameter_signature"] == "a-parameter-hash"


def test_checkpoint_version_mismatch_warns_and_loads(checkpoint_directory):
    """A checkpoint from another aeon version warns but is still restored."""
    path = checkpoint_directory / "checkpoint.pkl"
    _ready_estimator().save_checkpoint(path)
    _update_checkpoint_metadata(path, aeon_version="0.0.0")

    with pytest.warns(UserWarning, match="0.0.0") as record:
        restored = _Checkpointable.load_checkpoint(path)

    # losing a long fit to a patch release is worse than resuming with a warning
    assert restored.completed_ == [1, 2, 3]
    assert __version__ in str(record[0].message)


def test_read_checkpoint_metadata_inspects_unloadable(checkpoint_directory):
    """A checkpoint that cannot be loaded here can still be inspected."""
    path = checkpoint_directory / "checkpoint.pkl"
    _ready_estimator().save_checkpoint(path)
    _update_checkpoint_metadata(path, format_version=999, aeon_version="0.0.0")

    # refused by the loader, but the header still says what produced it
    with pytest.raises(ValueError, match="format version"):
        _Checkpointable.load_checkpoint(path)
    assert CheckpointableMixin.read_checkpoint_metadata(path)["aeon_version"] == "0.0.0"


def test_checkpoint_rejects_non_json_header(checkpoint_directory):
    """A file without a JSON header is refused before anything is unpickled."""
    path = checkpoint_directory / "checkpoint.pkl"
    # A pickle-first file is not a supported checkpoint layout.
    with path.open("wb") as file:
        pickle.dump({"format_version": 1}, file)
        pickle.dump(_ready_estimator(), file)

    with patch("aeon.base._checkpoint.pickle.load") as load:
        with pytest.raises(ValueError, match="not readable JSON"):
            _Checkpointable.load_checkpoint(path)
        load.assert_not_called()


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
        with patch("aeon.base._checkpoint.time.monotonic", return_value=200):
            estimator._start_checkpoint_timer()
        save.reset_mock()
        with patch("aeon.base._checkpoint.time.monotonic", return_value=250):
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
