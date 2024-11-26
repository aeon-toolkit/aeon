"""Interface compliance checkers for aeon estimators."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "parametrize_with_checks",
    "check_estimator",
]

import re
from functools import partial, wraps
from inspect import isclass
from typing import Callable, Optional, Union

from sklearn import config_context
from sklearn.utils._testing import SkipTest

from aeon.base import BaseAeonEstimator
from aeon.testing.estimator_checking._yield_estimator_checks import (
    _yield_all_aeon_checks,
)
from aeon.testing.testing_config import (
    EXCLUDE_ESTIMATORS,
    EXCLUDED_TESTS,
    EXCLUDED_TESTS_NO_NUMBA,
    NUMBA_DISABLED,
)
from aeon.utils.validation._dependencies import (
    _check_estimator_deps,
    _check_soft_dependencies,
)


def parametrize_with_checks(
    estimators: list[Union[BaseAeonEstimator, type[BaseAeonEstimator]]],
    use_first_parameter_set: bool = False,
) -> Callable:
    """Pytest specific decorator for parametrizing aeon estimator checks.

    The `id` of each check is set to be the name of the check with its keyword
    arguments, including a pprint version of the estimator.

    This allows to use `pytest -k` to specify which tests to run i.e.
        pytest -k check_fit_updates_state

    Based on the `scikit-learn``parametrize_with_checks` function.

    Parameters
    ----------
    estimators : list of aeon BaseAeonEstimator instances or classes
        Estimators to generate checks for. If an item is a class, an instance will
        be created using BaseAeonEstimator._create_test_instance().
    use_first_parameter_set : bool, default=False
        If True, only the first parameter set from _get_test_params will be used if a
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
    >>> from aeon.regression.interval_based import TimeSeriesForestRegressor
    >>> @parametrize_with_checks(
    ...                     [TimeSeriesForestClassifier, TimeSeriesForestRegressor])
    ... def test_aeon_compatible_estimator(check):
    ...     check()
    """
    _check_soft_dependencies("pytest")

    import pytest

    checks = []
    for est in estimators:
        # check if estimator has soft dependencies installed
        has_dependencies = _check_estimator_deps(est, severity="none")

        # collect all relevant checks
        for check in _yield_all_aeon_checks(
            est,
            use_first_parameter_set=use_first_parameter_set,
            has_dependencies=has_dependencies,
        ):
            # wrap check to skip if necessary (missing dependencies, on an exclude list
            # etc.)
            checks.append(_check_if_xfail(est, check, has_dependencies))

    # return a pytest parametrize decorator with custom ids
    return pytest.mark.parametrize(
        "check",
        checks,
        ids=_get_check_estimator_ids,
    )


def check_estimator(
    estimator: Union[BaseAeonEstimator, type[BaseAeonEstimator]],
    raise_exceptions: bool = False,
    use_first_parameter_set: bool = False,
    checks_to_run: Optional[Union[str, list[str]]] = None,
    checks_to_exclude: Optional[Union[str, list[str]]] = None,
    full_checks_to_run: Optional[Union[str, list[str]]] = None,
    full_checks_to_exclude: Optional[Union[str, list[str]]] = None,
    verbose: bool = False,
):
    """Check if estimator adheres to `aeon` conventions.

    This function will run an extensive test-suite to make sure that the estimator
    complies with `aeon` conventions.
    The checks run will differ based on the estimator input. There is a set of
    general checks for all estimators, and module-specific tests for classifiers,
    anomaly detectors, transformer, etc.
    Some checks may be skipped if the estimator has certain tags i.e.
    `non_deterministic`.

    Parameters
    ----------
    estimator : aeon BaseAeonEstimator instance or class
        Estimator to run checks on. If estimator is a class, an instance will
        be created using BaseAeonEstimator._create_test_instance().
    raise_exceptions : bool, optional, default=False
        Whether to return exceptions/failures in the results dict, or raise them
            if False: returns exceptions in returned `results` dict
            if True: raises exceptions as they occur
    use_first_parameter_set : bool, default=False
        If True, only the first parameter set from _get_test_params will be used if a
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
        name of the check to run with parameterization, i.e.
        "check_clone(estimator=MockClassifier())" or
        "check_fit_updates_state(estimator=MockClassifier())".

        Checks not passed will be excluded from testing. If None, all checks are run
        (unless excluded elsewhere).
    full_checks_to_exclude : str or list of str, default=None
        Full check name string(s) of checks to exclude. This should include the
        function name of the check to exclude with parameterization, i.e.
        "check_clone(estimator=MockClassifier())" or
        "check_fit_updates_state(estimator=MockClassifier())".

        If None, no checks are excluded (unless excluded elsewhere).
    verbose : str, optional, default=False.
        Whether to print out informative summary of tests run.

    Returns
    -------
    results : dict of test results
        The test results. Keys are parameterized check strings. The `id` of each check
        is set to be the name of the check with its keyword arguments, including a
        pprint version of the estimator.

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
    >>> check_estimator(MockClassifier, checks_to_run="check_get_params")
    {'check_get_params(estimator=MockClassifier())': 'PASSED'}
    """
    # check if estimator has soft dependencies installed
    _check_soft_dependencies("pytest")
    _check_estimator_deps(estimator)

    checks = []
    # collect all relevant checks
    for check in _yield_all_aeon_checks(
        estimator,
        use_first_parameter_set=use_first_parameter_set,
        has_dependencies=True,
    ):
        # wrap check to skip if necessary (on an exclude list etc.)
        checks.append(_check_if_skip(estimator, check, True))

    # process run/exclude lists to filter checks
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
    # run all checks
    for check in checks:
        check_name = _get_check_estimator_ids(check)

        # ignore check if filtered
        if checks_to_run is not None and check_name.split("(")[0] not in checks_to_run:
            continue
        if (
            checks_to_exclude is not None
            and check_name.split("(")[0] in checks_to_exclude
        ):
            continue
        if full_checks_to_run is not None and check_name not in full_checks_to_run:
            continue
        if full_checks_to_exclude is not None and check_name in full_checks_to_exclude:
            continue

        # run the check and process output/errors
        try:
            check()
            if verbose:
                print(f"PASSED: {name}")  # noqa T001
            results[check_name] = "PASSED"
            passed += 1
        except SkipTest as skip:
            if verbose:
                print(f"SKIPPED: {name}")  # noqa T001
            results[check_name] = "SKIPPED: " + str(skip)
            skipped += 1
        except Exception as exception:
            if raise_exceptions:
                raise exception
            elif verbose:
                print(f"FAILED: {name}")  # noqa T001
            results[check_name] = "FAILED: " + str(exception)
            failed += 1

    if verbose:
        print(  # noqa T001
            f"Tests run: {passed + skipped + failed}, Passed: {passed}, "
            f"Failed: {failed}, Skipped: {skipped}"
        )

    return results


def _check_if_xfail(estimator, check, has_dependencies):
    """Check if a check should be xfailed."""
    import pytest

    skip, reason, _ = _should_be_skipped(estimator, check, has_dependencies)
    if skip:
        return pytest.param(check, marks=pytest.mark.xfail(reason=reason))

    return check


def _check_if_skip(estimator, check, has_dependencies):
    """Check if a check should be skipped by raising a SkipTest exception."""
    skip, reason, check_name = _should_be_skipped(estimator, check, has_dependencies)
    if skip:

        @wraps(check)
        def wrapped(*args, **kwargs):
            est_name = (
                estimator.__name__
                if isclass(estimator)
                else estimator.__class__.__name__
            )
            raise SkipTest(f"Skipping {check_name} for {est_name}: {reason}")

        return wrapped
    return check


def _should_be_skipped(estimator, check, has_dependencies):
    est_name = (
        estimator.__name__ if isclass(estimator) else estimator.__class__.__name__
    )
    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

    # check estimator dependencies
    if not has_dependencies and "softdep" not in check_name:
        return True, "Incompatible dependencies or Python version", check_name

    # check aeon exclude lists
    if est_name in EXCLUDE_ESTIMATORS:
        return True, "In aeon estimator exclude list", check_name
    elif check_name in EXCLUDED_TESTS.get(est_name, []):
        return True, "In aeon test exclude list for estimator", check_name
    elif NUMBA_DISABLED and check_name in EXCLUDED_TESTS_NO_NUMBA.get(est_name, []):
        return True, "In aeon no numba test exclude list for estimator", check_name

    return False, "", check_name


def _get_check_estimator_ids(obj):
    """Create pytest ids for aeon checks.

    When `obj` is an estimator, this returns the sklearn pprint version of the
    estimator (with `print_changed_only=True`). When `obj` is a function, the
    name of the function is returned with its keyword arguments.

    `_get_check_estimator_ids` is designed to be used as the `id` in
    `pytest.mark.parametrize` where `checks_generator` is yielding estimators and
    checks.

    Some parameters which contain functions or methods will be obfuscated to
    allow for compatability with `pytest-xdist`. This requires that IDs on each thread
    be the same, and functions can generate different IDs.

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

        kwlist = []
        for k, v in obj.keywords.items():
            v = _get_check_estimator_ids(v)
            if v is not None:
                kwlist.append(f"{k}={v}")
        kwstring = ",".join(kwlist) if kwlist else ""
        return f"{obj.func.__name__}({kwstring})"
    elif isclass(obj):
        return obj.__name__
    elif hasattr(obj, "get_params"):
        with config_context(print_changed_only=True):
            s = re.sub(r"\s", "", str(obj))
            s = re.sub(r"<function[^)]*>", "func", s)
            s = re.sub(r"<boundmethodrv[^)]*>", "boundmethod", s)
            return s
    elif isinstance(obj, str):
        return obj
    else:
        return None
