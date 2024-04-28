"""Interface compliance checkers for aeon estimators."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "parametrize_with_checks",
    "check_estimator",
]

import re
from functools import partial, wraps
from inspect import isclass
from typing import Type, List, Callable, Union

from sklearn import config_context
from sklearn.utils._testing import SkipTest

from aeon.base import BaseEstimator
from aeon.testing.estimator_checks._yield_estimator_checks import _yield_all_aeon_checks

from aeon.testing.test_config import EXCLUDED_TESTS
from aeon.utils.validation._dependencies import _check_soft_dependencies


def parametrize_with_checks(estimators: List[Union[BaseEstimator, Type[BaseEstimator]]]) -> Callable:
    """Pytest specific decorator for parametrizing aeon estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.

    This allows to use `pytest -k` to specify which tests to run::
        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Based on the `scikit-learn``parametrize_with_checks` function.

    Parameters
    ----------
    estimators : list of aeon aseEstimator instances or classesB
        Estimators to generated checks for. If item is a class, an instance will
        be created using BaseEstimator.create_test_instance().

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to tsml or scikit-learn conventions.

    Examples
    --------
    >>> from aeon.testing import parametrize_with_checks
    >>> from aeon.classification.interval_based import TimeSeriesForestClassifier
    >>> from aeon.forecasting.naive import NaiveForecaster
    >>> @parametrize_with_checks([TimeSeriesForestClassifier, NaiveForecaster])
    ... def test_aeon_compatible_estimator(estimator, check):
    ...     check(estimator)
    """
    _check_soft_dependencies("pytest")

    import pytest

    def checks_generator():
        for est in estimators:
            if isclass(est):
                if issubclass(est, BaseEstimator):
                    est = est.create_test_instance()
                else:
                    raise TypeError(
                        f"Passed class {est} is not a subclass of BaseEstimator."
                    )
            elif not isinstance(est, BaseEstimator):
                raise TypeError(
                    f"Passed object {est} is not an instance of BaseEstimator."
                )

            name = type(est).__name__
            for check in _yield_all_aeon_checks(est):
                check = partial(check, name)
                yield _check_if_xfail(est, check)

    return pytest.mark.parametrize(
        "estimator, check",
        checks_generator(),
        ids=_get_check_estimator_ids,
    )


def check_estimator(
    estimator: Union[BaseEstimator, Type[BaseEstimator]],
    raise_exceptions=False,
    tests_to_run=None,
    fixtures_to_run=None,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
    verbose=True,
):
    """Check if estimator adheres to scikit-learn conventions.

    This function will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `scikit-learn`
    conventions as detailed in :ref:`rolling_your_own_estimator`.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    Setting `generate_only=True` returns a generator that yields (estimator,
    check) tuples where the check can be called independently from each
    other, i.e. `check(estimator)`. This allows all checks to be run
    independently and report the checks that are failing.

    scikit-learn provides a pytest specific decorator,
    :func:`~sklearn.utils.estimator_checks.parametrize_with_checks`, making it
    easier to test multiple estimators.

    Parameters
    ----------
    estimator : BaseEstimator instance or classe
        Estimator instance or class to check.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> check_estimator(LogisticRegression(), generate_only=True)
    <generator object ...>

    Run all tests on one single estimator.

    Tests that are run on estimator:
        all tests in test_all_estimators
        all interface compatibility tests from the module of estimator's type
            for example, test_all_forecasters if estimator is a forecaster

    Parameters
    ----------
    estimator : estimator class or estimator instance
    raise_exceptions : bool, optional, default=False
        whether to return exceptions/failures in the results dict, or raise them
            if False: returns exceptions in returned `results` dict
            if True: raises exceptions as they occur
    tests_to_run : str or list of str, optional. Default = run all tests.
        Names (test/function name string) of tests to run.
        sub-sets tests that are run to the tests given here.
    fixtures_to_run : str or list of str, optional. Default = run all tests.
        pytest test-fixture combination codes, which test-fixture combinations to run.
        sub-sets tests and fixtures to run to the list given here.
        If both tests_to_run and fixtures_to_run are provided, runs the *union*,
        i.e., all test-fixture combinations for tests in tests_to_run,
            plus all test-fixture combinations in fixtures_to_run.
    verbose : str, optional, default=True.
        whether to print out informative summary of tests run.
    tests_to_exclude : str or list of str, names of tests to exclude. default = None
        removes tests that should not be run, after subsetting via tests_to_run.
    fixtures_to_exclude : str or list of str, fixtures to exclude. default = None
        removes test-fixture combinations that should not be run.
        This is done after subsetting via fixtures_to_run.

    Returns
    -------
    results : dict of results of the tests in self
        keys are test/fixture strings, identical as in pytest, e.g., test[fixture]
        entries are the string "PASSED" if the test passed,
            or the exception raised if the test did not pass
        returned only if all tests pass,

    Raises
    ------
    raises any exception produced by the tests directly

    Examples
    --------
    >>> from aeon.transformations.exponent import ExponentTransformer
    >>> from aeon.testing.estimator_checks import check_estimator

    Running all tests for ExponentTransformer class,
    this uses all instances from get_test_params and compatible scenarios
    >>> results = check_estimator(ExponentTransformer)
    All tests PASSED!

    Running all tests for a specific ExponentTransformer
    this uses the instance that is passed and compatible scenarios
    >>> results = check_estimator(ExponentTransformer(42))
    All tests PASSED!

    Running specific test (all fixtures) for ExponentTransformer
    >>> results = check_estimator(ExponentTransformer, tests_to_run="test_clone")
    All tests PASSED!

    {'test_clone[ExponentTransformer-0]': 'PASSED',
    'test_clone[ExponentTransformer-1]': 'PASSED'}

    Running one specific test-fixture-combination for ExponentTransformer
    >>> check_estimator(
    ...    ExponentTransformer, fixtures_to_run="test_clone[ExponentTransformer-1]"
    ... )
    All tests PASSED!
    {'test_clone[ExponentTransformer-1]': 'PASSED'}
    """
    def checks_generator():
        est = estimator
        if isclass(est):
            if issubclass(est, BaseEstimator):
                est = est.create_test_instance()
            else:
                raise TypeError(
                    f"Passed class {est} is not a subclass of BaseEstimator."
                )
        elif not isinstance(est, BaseEstimator):
            raise TypeError(
                f"Passed object {est} is not an instance of BaseEstimator."
            )

        name = type(est).__name__
        for check in _yield_all_aeon_checks(est):
            check = partial(check, name)
            yield _check_if_skip(est, check)

    passed = 0
    skipped = 0
    failed = 0
    results = {}
    for est, check in checks_generator():
        name = _get_check_estimator_ids(check)
        try:
            check(est)
            if verbose:
                print(f"PASSED: {check_name}[{est_name}]")  # noqa T001
            results[name] = "PASSED"
            passed += 1
        except SkipTest as skip:
            if verbose:
                print(f"SKIPPED: {check_name}[{est_name}]")  # noqa T001
            results[name] = "SKIPPED: " + str(skip)
            skipped += 1
        except Exception as exception:
            if raise_exceptions:
                raise exception
            elif verbose:
                print(f"FAILED: {check_name}[{est_name}]")  # noqa T001
            results[name] = "FAILED: " + str(exception)
            failed += 1

    if verbose:
        print(
            f"Tests run: {passed + skipped + failed}, Passed: {passed}, "
            f"Failed: {failed}, Skipped: {skipped}"
        )


def _check_if_xfail(estimator, check):
    """Check if a check should be xfailed."""
    import pytest

    skip, reason = _should_be_skipped(estimator, check)
    if skip:
        return pytest.param(estimator, check, marks=pytest.mark.xfail(reason=reason))

    return estimator, check


def _check_if_skip(estimator, check):
    """Check if a check should be skipped by raising a SkipTest exception."""
    skip, reason = _should_be_skipped(estimator, check)
    if skip:
        check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

        @wraps(check)
        def wrapped(*args, **kwargs):
            raise SkipTest(
                f"Skipping {check_name} for {estimator.__class__.__name__}: {reason}"
            )

        return estimator, wrapped
    return estimator, check


def _should_be_skipped(estimator, check):
    return check.__name__ in EXCLUDED_TESTS.get(estimator.__name__, [])


def _get_check_estimator_ids(obj):
    """Create pytest ids for aeon checks.

    When `obj` is an estimator, this returns the sklearn pprint version of the
    estimator (with `print_changed_only=True`). When `obj` is a function, the
    name of the function is returned with its keyword arguments.

    `_get_check_estimator_ids` is designed to be used as the `id` in
    `pytest.mark.parametrize` where `checks_generator` is yielding estimators and
    checks.

    Based on the `scikit-learn` `_get_check_estimator_ids` function.

    Parameters
    ----------
    obj : estimator or function
        Items generated by `checks_generator`.

    Returns
    -------
    id : str or None
        The id of the check.
    """
    if callable(obj):
        if not isinstance(obj, partial):
            return obj.__name__

        if not obj.keywords:
            return obj.func.__name__

        kwstring = ",".join(["{}={}".format(k, v) for k, v in obj.keywords.items()])
        return "{}({})".format(obj.func.__name__, kwstring)
    elif hasattr(obj, "get_params"):
        with config_context(print_changed_only=True):
            s = re.sub(r"\s", "", str(obj))
            return re.sub(r"<function[^)]*>", "func", s)
    else:
        raise ValueError("Unexpected object: {}".format(obj))