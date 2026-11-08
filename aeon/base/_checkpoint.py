"""Mixin providing explicit checkpoint persistence for estimators."""

__all__ = ["CheckpointableMixin"]

import json
import os
import pickle
import tempfile
import time
import warnings
from numbers import Real
from pathlib import Path

import numpy as np
from joblib import hash as joblib_hash

from aeon import __version__

# Bump when the on-disk layout changes.
_CHECKPOINT_FORMAT_VERSION = 1


def _parse_checkpoint_header(file):
    """Parse the JSON header, leaving the handle at the start of the body.

    The format version is returned rather than checked, so callers can decide
    whether to inspect or to refuse an unsupported checkpoint.
    """
    try:
        metadata = json.loads(file.readline().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(
            "Checkpoint header is not readable JSON; the file is not an aeon "
            "checkpoint."
        ) from e
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint header is not a JSON object.")
    return metadata


class CheckpointableMixin:
    """Save and restore a complete estimator using Python pickle.

    Supporting estimators provide ``checkpoint_path`` and ``checkpoint_interval``
    constructor parameters, and call ``_checkpoint_if_due`` at safe boundaries.
    The interval is in minutes; None disables periodic saves. A configured path
    is also saved on successful completion. No background workers are started.

    Checkpoint timing is approximate. A periodic checkpoint is written at the
    next algorithm-specific safe boundary after the interval has elapsed.
    Long-running iterations or batches can delay writes beyond the interval.

    Subclasses must store all continuation state on the estimator and set
    ``_checkpoint_ready`` only after a complete, consistent update. Checkpointing
    must not run concurrently with fitting. Override save/load together if a
    different persistence backend is needed.

    A checkpoint is a single file holding a one-line JSON header followed by the
    pickled estimator. The header records the format and aeon versions, the
    estimator class and the parameter signature. The format version is checked
    before unpickling. The plain-text header can be inspected without loading
    the estimator; its class and parameter signature are informational.

    Only load trusted files: the body is pickle and can execute arbitrary code.
    The default implementation requires ``cant_pickle=False``. A checkpoint from
    a different aeon version raises ``UserWarning`` and is still
    loaded, following scikit-learn, because a version change usually leaves
    continuation state readable and refusing would discard the fit. Dependency
    versions and the Python environment should also be kept fixed.
    """

    _checkpoint_mutable_params = (
        "checkpoint_path",
        "checkpoint_interval",
        "n_jobs",
        "verbose",
    )

    def save_checkpoint(self, filepath):
        """Atomically save a resumable estimator to ``filepath``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file. Its parent directory must already exist.

        Notes
        -----
        A temporary file in the destination directory is flushed and synced,
        then replaces the destination. Failed serialization leaves the previous
        checkpoint intact. Concurrent writers to the same path are unsupported.
        """
        if self.get_tag("cant_pickle"):
            raise ValueError(
                "Default checkpoint persistence requires cant_pickle=False."
            )
        if not getattr(self, "_checkpoint_ready", False):
            raise ValueError("No safe continuation state is available to checkpoint.")
        filepath = Path(filepath)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=filepath.parent, prefix=f".{filepath.name}.", delete=False
            ) as file:
                temporary_path = Path(file.name)
                metadata = {
                    "format_version": _CHECKPOINT_FORMAT_VERSION,
                    "aeon_version": __version__,
                    "estimator_class": self._checkpoint_class_name(),
                    "parameter_signature": getattr(
                        self, "_checkpoint_parameter_signature", None
                    ),
                }
                # json.dumps escapes newlines, so the header is always one line
                file.write(json.dumps(metadata).encode("utf-8") + b"\n")
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, filepath)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, filepath):
        """Load an estimator checkpoint from ``filepath``.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Source checkpoint file. Only load files from trusted sources.

        Returns
        -------
        estimator : CheckpointableMixin
            The restored estimator.

        Raises
        ------
        TypeError
            If the checkpoint does not contain an instance of ``cls``.

        Warns
        -----
        UserWarning
            If the checkpoint was written by a different aeon version. The
            estimator is still loaded.
        """
        with Path(filepath).open("rb") as file:
            # Reject unreadable headers and unsupported formats before unpickling.
            metadata = _parse_checkpoint_header(file)
            if metadata.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
                raise ValueError("Unsupported checkpoint format version.")
            if metadata.get("aeon_version") != __version__:
                # a version change usually leaves continuation state readable, so
                # warn and let the caller decide rather than discarding the work
                warnings.warn(
                    f"Trying to load a {metadata.get('estimator_class')} checkpoint "
                    f"written by aeon {metadata.get('aeon_version')} while running "
                    f"aeon {__version__}. Continuation state is restored by pickle, "
                    "so an estimator whose attributes have changed between these "
                    "versions may resume incorrectly or fail. Resume in the "
                    "environment that wrote the checkpoint where possible.",
                    UserWarning,
                    stacklevel=2,
                )
            estimator = pickle.load(file)

        if not isinstance(estimator, cls):
            raise TypeError(
                f"Checkpoint contains {type(estimator).__name__}, "
                f"expected an instance of {cls.__name__}."
            )

        return estimator

    @staticmethod
    def read_checkpoint_metadata(filepath):
        """Read a checkpoint's header without loading the estimator.

        Nothing in the file is unpickled, so this is safe to run on checkpoints
        that would be refused by ``load_checkpoint``, or that are not trusted
        enough to load at all.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Source checkpoint file.

        Returns
        -------
        metadata : dict
            The header as stored, with at least ``format_version``,
            ``aeon_version``, ``estimator_class`` and ``parameter_signature``.
            The format version is reported rather than enforced, so a
            checkpoint this version of aeon cannot load can still be inspected.

        Raises
        ------
        ValueError
            If the file does not start with a readable JSON header.

        Examples
        --------
        >>> from aeon.base import CheckpointableMixin  # doctest: +SKIP
        >>> CheckpointableMixin.read_checkpoint_metadata("run.pkl")  # doctest: +SKIP
        {'format_version': 1, 'aeon_version': '1.6.0', ...}
        """
        with Path(filepath).open("rb") as file:
            return _parse_checkpoint_header(file)

    @classmethod
    def _checkpoint_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def _init_checkpoint(self, X, y):
        self._start_checkpoint_timer()
        self._checkpoint_data_signature = joblib_hash((X, y), coerce_mmap=True)
        self._checkpoint_parameter_signature = self._checkpoint_parameter_hash()
        self._checkpoint_ready = False

    def _checkpoint_parameter_hash(self):
        # a generator passed as random_state is consumed by fitting, by the
        # estimator or by components sharing it, so it is identified by type:
        # its evolving state is not a change of model configuration
        return joblib_hash(
            {
                name: (
                    type(value) if isinstance(value, np.random.RandomState) else value
                )
                for name, value in self.get_params(deep=False).items()
                if name not in self._checkpoint_mutable_params
            }
        )

    def _validate_checkpoint(self, X, y):
        if not getattr(self, "_checkpoint_ready", False):
            raise ValueError("No safe continuation state is available; call fit first.")
        if joblib_hash((X, y), coerce_mmap=True) != self._checkpoint_data_signature:
            raise ValueError(
                "Unable to resume fitting: X and y must match the training data "
                "used in the original fit."
            )
        if self._checkpoint_parameter_hash() != self._checkpoint_parameter_signature:
            raise ValueError("Model-building parameters have changed since fitting.")

    def _start_checkpoint_timer(self):
        """Validate checkpoint settings and restart timing for fit or resume_fit.

        The interval must be None (final-only saving) or positive finite minutes.
        When a destination is configured, require an existing parent directory,
        a file destination and an estimator that supports pickling. Validate
        these settings before training starts, even if no periodic save is due.
        """
        interval = self.checkpoint_interval
        if interval is not None and (
            isinstance(interval, bool)
            or not isinstance(interval, Real)
            or not np.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError(
                "checkpoint_interval must be None or positive finite minutes."
            )
        if self.checkpoint_path is not None:
            path = Path(self.checkpoint_path)
            parent = path.parent
            if not parent.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint parent directory does not exist or is not a "
                    f"directory: {parent}"
                )
            if path.is_dir():
                raise IsADirectoryError(
                    f"Checkpoint path must be a file, not a directory: {path}"
                )
            if self.get_tag("cant_pickle"):
                raise ValueError("Default checkpointing requires cant_pickle=False.")
        # Each fit or resume_fit call starts a fresh checkpoint interval.
        self._checkpoint_last_time = time.monotonic()

    def _checkpoint_if_due(self, force=False):
        """Save at a safe boundary when due, or on successful completion."""
        if self.checkpoint_path is None:
            return
        if force or (
            self.checkpoint_interval is not None
            and time.monotonic() - self._checkpoint_last_time
            >= self.checkpoint_interval * 60
        ):
            self.save_checkpoint(self.checkpoint_path)
            self._checkpoint_last_time = time.monotonic()
