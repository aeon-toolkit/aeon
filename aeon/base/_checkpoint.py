"""Mixin providing explicit checkpoint persistence for estimators."""

__all__ = ["CheckpointableMixin"]

import os
import pickle
import tempfile
import time
from numbers import Real
from pathlib import Path

import numpy as np
from joblib import hash as joblib_hash

from aeon import __version__


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

    Only load trusted files: pickle can execute arbitrary code. The default
    implementation requires ``cant_pickle=False`` and the same aeon version.
    Dependency versions and the Python environment should also be kept fixed.
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
                pickle.dump(
                    {
                        "format_version": 1,
                        "aeon_version": __version__,
                        "estimator_class": self._checkpoint_class_name(),
                    },
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
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
        """
        with Path(filepath).open("rb") as file:
            metadata = pickle.load(file)
            if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
                raise ValueError("Unsupported checkpoint format version.")
            if metadata.get("aeon_version") != __version__:
                raise ValueError(
                    "Checkpoint aeon version does not match this environment."
                )
            estimator = pickle.load(file)

        if not isinstance(estimator, cls):
            raise TypeError(
                f"Checkpoint contains {type(estimator).__name__}, "
                f"expected an instance of {cls.__name__}."
            )

        if metadata.get("estimator_class") != estimator._checkpoint_class_name():
            raise ValueError("Checkpoint estimator class does not match its metadata.")
        if not getattr(estimator, "_checkpoint_ready", False):
            raise ValueError("Checkpoint does not contain safe continuation state.")
        return estimator

    @classmethod
    def _checkpoint_class_name(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    def _init_checkpoint(self, X, y):
        self._start_checkpoint_timer()
        self._checkpoint_data_signature = joblib_hash((X, y), coerce_mmap=True)
        self._checkpoint_parameter_signature = self._checkpoint_parameter_hash()
        self._checkpoint_ready = False

    def _checkpoint_parameter_hash(self):
        return joblib_hash(
            {
                name: value
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
            Path(self.checkpoint_path)
            if self.get_tag("cant_pickle"):
                raise ValueError("Default checkpointing requires cant_pickle=False.")
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
