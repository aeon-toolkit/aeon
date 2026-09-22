# Base

The `aeon.base` module contains abstract base classes.

## Base classes

```{eval-rst}
.. currentmodule:: aeon.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseAeonEstimator
    BaseCollectionEstimator
    BaseSeriesEstimator
    CheckpointableMixin
    ComposableEstimatorMixin
```

## Checkpointing

Estimators declaring `capability:checkpointing=True` can save training state
and explicitly continue it. Initially, Arsenal supports this capability.
`fit` always starts a fresh fit; use `resume_fit` to continue a loaded checkpoint.

```python
from aeon.classification.convolution_based import Arsenal

clf = Arsenal(
    n_estimators=100,
    checkpoint_path="arsenal.pkl",
    checkpoint_interval=30,
    random_state=0,
)
clf.fit(X_train, y_train)

# In a later job, supply the original training data in the original order.
clf = Arsenal.load_checkpoint("arsenal.pkl")
clf.resume_fit(X_train, y_train)
```

The interval is in minutes. Writes occur after a complete batch has joined and
its results and RNG state have been committed. A successful fit also writes a
final checkpoint. With `checkpoint_interval=None`, only the final automatic
write occurs; with `checkpoint_path=None`, automatic writes are disabled.
Manual `clf.save_checkpoint(path)` is available after fitting. A killed job
loses work since the last completed write, and no checkpoint exists until the
first write. Set the interval shorter than the job's wall-time allowance.

For contracted Arsenal fits, `time_limit_in_minutes` is a fresh budget for each
invocation. For example, set `clf.time_limit_in_minutes = 3500` before resuming
to allow another 3500 minutes. `contract_max_n_estimators` remains a limit on
the total ensemble size; increase it if it has already been reached. A batch
may overrun the time budget. `fit_elapsed_time_` reports accumulated fitting
time separately. Non-contracted fits may continue after increasing `n_estimators`.

Continuation validates a joblib hash of the converted training data and labels,
including values, order and dtypes. This reads the dataset once per fit/resume
and does not store a second copy. It rejects a different fold even if shapes
match. Except for member limits, time budgets, worker counts, verbosity and
checkpoint settings, parameters must remain unchanged. OOB training-estimate
state is preserved when fitting began with `fit_predict` or `fit_predict_proba`;
`resume_fit` returns the estimator, not training predictions.

Persistence uses whole-estimator pickle and requires `cant_pickle=False`.
Only load trusted checkpoints. Files record the aeon version, estimator class
and checkpoint format version; incompatible aeon or format versions are rejected.
Use the same Python and dependency environment when resuming; cross-version
compatibility is not guaranteed, including between development revisions with
the same version number.

Writes use a temporary file in the destination directory, followed by flush,
file sync and atomic replacement. Failed serialization preserves the previous
checkpoint. The parent directory must exist and the filesystem must support
atomic replacement. Do not share a checkpoint path between concurrent jobs or
save while fitting in another thread. Scheduler termination during a write can
leave a temporary file; the last completed checkpoint remains the recovery file.

Implementers should inherit `CheckpointableMixin`, set the capability tag,
provide `checkpoint_path` and `checkpoint_interval` constructor parameters,
and implement `_resume_fit`. All continuation state must live on the estimator.
Set `_checkpoint_ready=False` before a batch and set it to `True` only after
committing results, counters and RNG state; then call `_maybe_checkpoint()`.
The classifier base handles data signatures and final writes. Provide a small
deterministic `checkpointing` test parameter set that reaches multiple safe
boundaries for the generic interruption/recovery estimator check. Add specific
tests for algorithmic counters, random state and retained training estimates.
