"""Interface compliance checkers for aeon estimators."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "parametrize_with_checks",
    "check_estimator",
]

import re
from functools import partial, wraps
from inspect import isclass
from typing import Callable, List, Type, Union

from sklearn import config_context
from sklearn.utils._testing import SkipTest

from aeon.base import BaseEstimator
from aeon.testing.estimator_checking._yield_estimator_checks import (
    _yield_all_aeon_checks,
)
from aeon.testing.test_config import EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
from aeon.utils.validation._dependencies import (
    _check_estimator_deps,
    _check_soft_dependencies,
)


def parametrize_with_checks(
    estimators: List[Union[BaseEstimator, Type[BaseEstimator]]],
    use_first_parameter_set: bool = False,
) -> Callable:
    """Pytest specific decorator for parametrizing aeon estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.

    This allows to use `pytest -k` to specify which tests to run i.e.
        pytest -k check_fit_updates_state

    Based on the `scikit-learn``parametrize_with_checks` function.

    Parameters
    ----------
    estimators : list of aeon BaseEstimator instances or classes
        Estimators to generate checks for. If an item is a class, an instance will
        be created using BaseEstimator.create_test_instance().
    use_first_parameter_set : bool, default=False
        If True, only the first parameter set from get_test_params will be used if a
        class is passed.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to tsml or scikit-learn conventions.

    Examples
    --------
    >>> from aeon.testing.estimator_checking import parametrize_with_checks
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
                    est = est.create_test_instance(return_first=use_first_parameter_set)
                else:
                    raise TypeError(
                        f"Passed class {est} is not a subclass of BaseEstimator."
                    )
            elif not isinstance(est, BaseEstimator):
                raise TypeError(
                    f"Passed object {est} is not an instance of BaseEstimator."
                )

            if not isinstance(est, list):
                est = [est]

            for e in est:
                for check in _yield_all_aeon_checks(e):
                    yield _check_if_xfail(e, check)

    return pytest.mark.parametrize(
        "estimator, check",
        checks_generator(),
        ids=_get_check_estimator_ids,
    )


def check_estimator(
    estimator: Union[BaseEstimator, Type[BaseEstimator]],
    raise_exceptions: bool = False,
    use_first_parameter_set: bool = False,
    checks_to_run: Union[str, List[str]] = None,
    checks_to_exclude: Union[str, List[str]] = None,
    full_checks_to_run: Union[str, List[str]] = None,
    full_checks_to_exclude: Union[str, List[str]] = None,
    verbose: bool = False,
):
    """Check if estimator adheres to `aeon` conventions.

    This function will run an extensive test-suite to make sure that the estimator
    complies with `aeon` conventions.
    The checks run will differ based on the estimator input. There is a set of
    general checks for all estimators, and module-specific tests for classifiers,
    anomaly detectors, transformer, etc.
    Some checks may be skipped if the estimator has certain tags i.e.
    `non-deterministic`.

    Parameters
    ----------
    estimator : aeon BaseEstimator instances or classes
        Estimator to run checks on. If estimator is a class, an instance will
        be created using BaseEstimator.create_test_instance().
    raise_exceptions : bool, optional, default=False
        Whether to return exceptions/failures in the results dict, or raise them
            if False: returns exceptions in returned `results` dict
            if True: raises exceptions as they occur
    use_first_parameter_set : bool, default=False
        If True, only the first parameter set from get_test_params will be used if a
        class is passed.
    checks_to_run : str or list of str, default=None
        Name(s) of checks to run. This should include the function name of the check to
        run without parameterization, i.e. "check_clone" or "check_fit_updates_state".

        Checks not passed will be excluded from testing. If None, all checks are run
        (unless excluded elsewhere).
    checks_to_exclude : str or list of str, default=None
        Name(s) of checks to exclude. This should include the function name of the
        check to exclude without parameterization, i.e. "check_clone" or
        "check_fit_updates_state".

        If None, no checks are excluded (unless excluded elsewhere).
    full_checks_to_run : str or list of str, default=None
        Full check name string(s) of checks to run. This should include the function
        name of the check to with parameterization, i.e. "MockClassifier()-check_clone"
        or "MockClassifier()-check_fit_updates_state".

        Checks not passed will be excluded from testing. If None, all checks are run
        (unless excluded elsewhere).
    full_checks_to_exclude : str or list of str, default=None
        Full check name string(s) of checks to exclude. This should include the
        function name of the check to exclude with parameterization, i.e.
        "MockClassifier()-check_clone" or "MockClassifier()-check_fit_updates_state"

        If None, no checks are excluded (unless excluded elsewhere).
    verbose : str, optional, default=False.
        Whether to print out informative summary of tests run.

    Returns
    -------
    results : dict of test results
        The test results. Keys are parameterized check strings. The `id` of each check
        is set to be a pprint version of the estimator and the name of the check with
        its keyword arguments.

        Entries are the string "PASSED" if the test passed, the exception raised if
        the test did not pass, or the reason for skipping the test.

        If `raise_exceptions` is True, this is only returned if all tests pass.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.

    Examples
    --------
    >>> from aeon.testing.mock_estimators import MockClassifier
    >>> from aeon.testing.estimator_checking import check_estimator

    Running all tests for MockClassifier class
    >>> results = check_estimator(MockClassifier)

    Running all tests for a MockClassifier instance
    >>> results = check_estimator(MockClassifier())

    Running specific check for MockClassifier
    >>> check_estimator(MockClassifier, checks_to_run="check_clone")
    {'MockClassifier()-check_clone': 'PASSED'}
    """
    _check_estimator_deps(estimator)

    def checks_generator():
        est = estimator
        if isclass(est):
            if issubclass(est, BaseEstimator):
                est = est.create_test_instance(return_first=use_first_parameter_set)
            else:
                raise TypeError(
                    f"Passed class {est} is not a subclass of BaseEstimator."
                )
        elif not isinstance(est, BaseEstimator):
            raise TypeError(f"Passed object {est} is not an instance of BaseEstimator.")

        if not isinstance(est, list):
            est = [est]

        for e in est:
            for check in _yield_all_aeon_checks(e):
                yield _check_if_skip(e, check)

    if not isinstance(checks_to_run, (list, tuple)) and checks_to_run is not None:
        checks_to_run = [checks_to_run]
    if (
        not isinstance(checks_to_exclude, (list, tuple))
        and checks_to_exclude is not None
    ):
        checks_to_exclude = [checks_to_exclude]
    if (
        not isinstance(full_checks_to_run, (list, tuple))
        and full_checks_to_run is not None
    ):
        full_checks_to_run = [full_checks_to_run]
    if (
        not isinstance(full_checks_to_exclude, (list, tuple))
        and full_checks_to_exclude is not None
    ):
        full_checks_to_exclude = [full_checks_to_exclude]

    passed = 0
    skipped = 0
    failed = 0
    results = {}
    for est, check in checks_generator():
        check_name = _get_check_estimator_ids(check)
        full_name = f"{_get_check_estimator_ids(est)}-{check_name}"

        if checks_to_run is not None and check_name.split("(")[0] not in checks_to_run:
            continue
        if (
            checks_to_exclude is not None
            and check_name.split("(")[0] in checks_to_exclude
        ):
            continue
        if full_checks_to_run is not None and full_name not in full_checks_to_run:
            continue
        if full_checks_to_exclude is not None and full_name in full_checks_to_exclude:
            continue

        try:
            check(est)
            if verbose:
                print(f"PASSED: {name}")  # noqa T001
            results[full_name] = "PASSED"
            passed += 1
        except SkipTest as skip:
            if verbose:
                print(f"SKIPPED: {name}")  # noqa T001
            results[full_name] = "SKIPPED: " + str(skip)
            skipped += 1
        except Exception as exception:
            if raise_exceptions:
                raise exception
            elif verbose:
                print(f"FAILED: {name}")  # noqa T001
            results[full_name] = "FAILED: " + str(exception)
            failed += 1

    if verbose:
        print(  # noqa T001
            f"Tests run: {passed + skipped + failed}, Passed: {passed}, "
            f"Failed: {failed}, Skipped: {skipped}"
        )

    return results


def _check_if_xfail(estimator, check):
    """Check if a check should be xfailed."""
    import pytest

    skip, reason, _ = _should_be_skipped(estimator, check)
    if skip:
        return pytest.param(estimator, check, marks=pytest.mark.xfail(reason=reason))

    return estimator, check


def _check_if_skip(estimator, check):
    """Check if a check should be skipped by raising a SkipTest exception."""
    skip, reason, name = _should_be_skipped(estimator, check)
    if skip:

        @wraps(check)
        def wrapped(*args, **kwargs):
            raise SkipTest(
                f"Skipping {name} for {estimator.__class__.__name__}: {reason}"
            )

        return estimator, wrapped
    return estimator, check


def _should_be_skipped(estimator, check):
    est_name = estimator.__class__.__name__

    # check estimator dependencies
    if not _check_estimator_deps(estimator, severity=None):
        return True, "Incompatible dependencies or Python version"

    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

    # check aeon exclude lists
    if est_name in EXCLUDE_ESTIMATORS:
        return True, "In aeon estimator exclude list"
    elif check_name in EXCLUDED_TESTS.get(est_name, []):
        return True, "In aeon test exclude list for estimator"

    return False, "", check_name


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

        kwstring = ",".join([f"{k}={v}" for k, v in obj.keywords.items()])
        return f"{obj.func.__name__}({kwstring})"
    elif hasattr(obj, "get_params"):
        with config_context(print_changed_only=True):
            s = re.sub(r"\s", "", str(obj))
            return re.sub(r"<function[^)]*>", "func", s)
    else:
        raise ValueError(f"Unexpected object: {obj}")
