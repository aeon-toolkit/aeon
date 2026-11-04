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
and explicitly continue it. Arsenal, DrCIFClassifier, ShapeletTransformClassifier
and TemporalDictionaryEnsemble support this capability. The scikit-learn facing
Rotation Forest estimators, `RotationForestClassifier` and
`RotationForestRegressor`, support it too, through the same `save_checkpoint`,
`load_checkpoint` and `resume_fit` methods; they carry no tags, so the
capability is documented rather than declared.
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

The interval is in minutes, and checkpoint timing is approximate. Periodic writes
occur at the next algorithm-specific safe boundary after the interval has elapsed.
For Arsenal, DrCIFClassifier and Rotation Forest, this is after a complete batch
has joined and its results and RNG state have been committed; for
TemporalDictionaryEnsemble, after an evaluated parameter combination has been
retained or discarded. A long-running batch or evaluation can therefore delay
checkpoints beyond the requested interval. A TDE checkpoint holds the fitted
ensemble members, each keeping the word counts for its training subsample, so
files are as large as the fitted model. A successful fit also writes a
final checkpoint. With `checkpoint_interval=None`, only the final automatic
write occurs; with `checkpoint_path=None`, automatic writes are disabled.
Manual `clf.save_checkpoint(path)` is available after fitting. A killed job
loses work since the last completed write, and no checkpoint exists until the
first write. Set the interval shorter than the job's wall-time allowance.

For contracted Arsenal and DrCIFClassifier fits, `time_limit_in_minutes` bounds
total training time across calls, measured by `fit_elapsed_time_`. A fit
interrupted after 10 hours of a 12 hour contract resumes with 2 hours remaining; raise
`time_limit_in_minutes` above the time already spent to grant more. `fit` always
starts the budget afresh. `contract_max_n_estimators` remains a limit on the
total ensemble size; increase it if it has already been reached. A batch may
overrun the time budget.

`fit_time_millis_` records the initial fit call and is not updated by
`resume_fit`. Automatic checkpoints may omit it because they are saved before
the fit timer is assigned. Use `fit_elapsed_time_` for cumulative
contract timing across fit and resume calls.

Rotation Forest behaves as Arsenal does, at batches of `n_jobs` trees, with
`n_estimators` and `contract_max_n_estimators` as the mutable limits and
`fit_elapsed_time_` measuring the contract across calls. A forest fitted through
`fit_predict` or `fit_predict_proba` keeps the per-tree transformed training
data its out-of-bag estimates are built from, so those checkpoints are
considerably larger than the plain `fit` ones.

ShapeletTransformClassifier has two phases and a safe boundary between them. The
shapelet transform has none inside it, so it is all or nothing: a checkpoint is
written as soon as it completes, whatever the interval, because repeating it is
the cost a checkpoint exists to avoid. The estimator fit that follows is
resumable when the estimator supports checkpointing, as the default
`RotationForestClassifier` does. A nested estimator holds no path of its own;
its boundaries save the classifier containing it, so a pipeline is one
checkpoint file rather than several. With any other estimator, a resumed fit
rebuilds it from the start on the saved transform. Only `time_limit_in_minutes`
and the runtime settings may change between calls, since everything else shapes
the finished transform or the estimator a continued fit inherits. `resume_fit`
returns the classifier, so a fit begun with `fit_predict` or `fit_predict_proba`
and then resumed gives no train estimates.

`resume_fit` is not limited to recovering an interrupted fit: because member
limits apply to the whole ensemble, it also resizes one that completed normally,
without needing a checkpoint file. Raising `n_estimators` (or
`contract_max_n_estimators`) builds the extra members and yields an ensemble
identical to an uninterrupted fit of that size, so an ensemble can be grown
cheaply as more compute becomes available. Lowering it discards the newest
members and keeps the oldest; the random generators are not rewound, so raising
the limit again trains new members rather than restoring the discarded ones.
Growing a contracted fit also requires raising `time_limit_in_minutes` above
`fit_elapsed_time_`, since the contract bounds total training time.

A `random_state` holding a `RandomState` instance is identified in that
signature by its type rather than its value, because fitting consumes it, either
directly or through components sharing it, and a generator that has moved on is
not a change of model configuration.

TemporalDictionaryEnsemble continues its guided parameter search instead, so the
mutable limits are `n_parameter_samples` and `contract_max_n_parameter_samples`,
counting every combination evaluated across calls. `max_ensemble_size` cannot
change: members are retained and replaced against it as each candidate is
evaluated, so a later value would neither recover discarded candidates nor
reproduce an uninterrupted fit. A candidate whose evaluation fails is repeated
on resuming rather than skipped, since its parameters stay in the search space
and the random state is committed only once it has been evaluated. TDE gives its
members `random_state` itself, so with a `RandomState` instance the members built
after resuming depend on anything that consumed that generator in between,
predictions from the checkpoint included; pass an integer seed where a resumed
fit must match an uninterrupted one exactly.

Continuation validates a joblib hash of the converted training data and labels,
including values, order and dtypes. This reads the dataset once per fit/resume
and does not store a second copy. It rejects a different fold even if shapes
match. Except for member limits, time budgets, worker counts, verbosity and
checkpoint settings, parameters must remain unchanged. OOB training-estimate
state is preserved when fitting began with `fit_predict` or `fit_predict_proba`;
`resume_fit` returns the estimator, not training predictions.

Persistence uses whole-estimator pickle and requires `cant_pickle=False`.
Only load trusted checkpoints: the body is pickle and can execute arbitrary
code. Each file begins with a one-line JSON header recording the format version,
aeon version, estimator class and parameter signature. The format version is
checked before unpickling; the estimator class and parameter signature in the
header are informational. `CheckpointableMixin.read_checkpoint_metadata(path)` returns that
header without loading the estimator, so a checkpoint can be identified even
when it cannot be restored. An unsupported format version is rejected; a
checkpoint written by a different aeon version raises `UserWarning`
and is still loaded, since refusing it would discard the fit. Use the same
Python and dependency environment when resuming; cross-version compatibility is
not guaranteed, including between development revisions with the same version
number.

Writes use a temporary file in the destination directory, followed by flush,
file sync and atomic replacement. Failed serialization preserves the previous
checkpoint. The parent directory must exist and the filesystem must support
atomic replacement. Do not share a checkpoint path between concurrent jobs or
save while fitting in another thread. Scheduler termination during a write can
leave a temporary file; the last completed checkpoint remains the recovery file.

Implementers should inherit `CheckpointableMixin`, set the capability tag,
provide `checkpoint_path` and `checkpoint_interval` constructor parameters,
and implement `_resume_fit`. All continuation state must live on the estimator.
An estimator outside the aeon base classes, such as Rotation Forest, has no tag
system and provides its own public `resume_fit` instead. An estimator fitted
inside another calls `_set_checkpoint_parent`, which routes its boundaries to
the parent; a child boundary is a periodic opportunity only, never a forced
write, since the child finishing does not mean the parent has.
Override `_validate_resume_fit(X, y)` to reject invalid continuation parameters
before the fitted flag is cleared. This hook must not change fitted state.
Set `_checkpoint_ready=False` before a batch and set it to `True` only after
committing results, counters and RNG state; then call `_checkpoint_if_due()`.
The classifier base handles data signatures and final writes. Provide a small
deterministic `checkpointing` test parameter set that reaches multiple safe
boundaries for the generic interruption/recovery estimator check. Add specific
tests for algorithmic counters, random state and retained training estimates.

Interval forests inherit checkpoint persistence and continuation from
`BaseIntervalForest`. DrCIFClassifier enables this through its capability tag
and constructor parameters. Other interval forests do not yet expose this
capability. DrCIF checkpoints retain fitted representation transformers, trees,
interval selectors and RNG state. When fitting requests training estimates,
each batch also commits its OOB predictions before saving; transformed training
matrices are not retained for this purpose. Changing `parallel_backend` between
calls is allowed, as is changing `n_jobs`.
