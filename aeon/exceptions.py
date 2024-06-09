"""Custom exceptions and warnings."""

__all__ = ["NotEvaluatedError", "NotFittedError", "FitFailedWarning"]

from deprecated.sphinx import deprecated

# todo delete file when deprecated


# TODO: remove v0.10.0
@deprecated(
    version="0.9.0",
    reason="NotEvaluatedError will be removed in v0.10.0.",
    category=FutureWarning,
)
class NotEvaluatedError(ValueError, AttributeError):
    """NotEvaluatedError.

    Exception class to raise if evaluator is used before having evaluated any metric.
    """


# TODO: remove v0.10.0
@deprecated(
    version="0.9.0",
    reason="NotFittedError will be removed in v0.10.0. Use the sklearn "
    "version of the warning.",
    category=FutureWarning,
)
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    References
    ----------
    .. [1] Based on scikit-learn's NotFittedError
    """


# TODO: remove v0.10.0
@deprecated(
    version="0.9.0",
    reason="FitFailedWarning will be removed in v0.10.0. Use the sklearn "
    "version of the warning.",
    category=FutureWarning,
)
class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    FitFailedWarning('Estimator fit failed. The score on this train-test
    partition for these parameters will be set to 0.000000').

    References
    ----------
    .. [1] Based on scikit-learn's FitFailedWarning
    """
