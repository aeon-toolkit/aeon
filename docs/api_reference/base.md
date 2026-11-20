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
and continue it later. Arsenal and DrCIFClassifier support this capability.
`fit` always starts a fresh fit; `resume_fit` continues a loaded checkpoint, or
a completed fit, and `resume_fit_predict_proba` does the same for a fit begun
with `fit_predict_proba` and returns its train estimates.

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

`checkpoint_interval` is in minutes. Writes happen at the next safe boundary
after the interval has elapsed, and a successful fit also writes a final
checkpoint. A killed job loses the work done since the last write, so set the
interval shorter than the job's wall-time allowance. Any time contract covers
total training time across calls. Which parameters may change between calls is
documented by each estimator.

Resuming checks that the training data and labels match the original fit,
including their order. Checkpoints are pickled, so only load trusted files and
resume in the same Python and dependency environment.
`CheckpointableMixin.read_checkpoint_metadata(path)` reads a checkpoint's
header without unpickling it. See `CheckpointableMixin` for the file format and
how checkpoints are written.

Implementers should inherit `CheckpointableMixin`, set the capability tag,
provide `checkpoint_path` and `checkpoint_interval` constructor parameters,
and implement `_resume_fit`. All continuation state must live on the estimator.
Override `_validate_resume_fit(X, y)` to reject invalid continuation parameters
before the fitted flag is cleared. This hook must not change fitted state.
Set `_checkpoint_ready=False` before a batch and set it to `True` only after
committing results, counters and RNG state; then call `_checkpoint_if_due()`.
The classifier base handles data signatures and final writes. Provide a small
deterministic `checkpointing` test parameter set that reaches multiple safe
boundaries for the generic interruption/recovery estimator check. Add specific
tests for algorithmic counters, random state and retained training estimates.

Interval forests inherit checkpointing from `BaseIntervalForest`, and a subclass
enables it by setting the capability tag and exposing the constructor
parameters, as DrCIFClassifier does.
