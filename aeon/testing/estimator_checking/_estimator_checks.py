"""Interface compliance checkers for aeon estimators."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "parametrize_with_checks",
    "check_estimator",
]

import re
import warnings
from functools import partial, wraps
from inspect import isclass
from typing import Callable, List, Type, Union

from sklearn import config_context
from sklearn.utils._testing import SkipTest

from aeon.base import BaseEstimator
from aeon.forecasting.base import BaseForecaster
from aeon.testing.estimator_checking._legacy._legacy_estimator_checks import (
    check_estimator_legacy,
)
from aeon.testing.estimator_checking._yield_estimator_checks import (
    _yield_all_aeon_checks,
)
from aeon.testing.test_config import EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer
from aeon.utils.validation._dependencies import (
    _check_estimator_deps,
    _check_soft_dependencies,
)


def _is_legacy_estimator(estimator):
    if isclass(estimator):
        if issubclass(estimator, BaseForecaster) or (
            issubclass(estimator, BaseTransformer)
            and not (
                issubclass(estimator, BaseSeriesTransformer)
                or issubclass(estimator, BaseCollectionTransformer)
            )
        ):
            return True
    else:
        if isinstance(estimator, BaseForecaster) or (
            isinstance(estimator, BaseTransformer)
            and not (
                isinstance(estimator, BaseSeriesTransformer)
                or isinstance(estimator, BaseCollectionTransformer)
            )
        ):
            return True
    return False


def parametrize_with_checks(
    estimators: List[Union[BaseEstimator, Type[BaseEstimator]]],
    use_first_parameter_set: bool = False,
) -> Callable:
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
            if _is_legacy_estimator(est):
                continue

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
    checks_to_run: List[str] = None,
    checks_to_exclude: List[str] = None,
    full_checks_to_run: List[str] = None,
    full_checks_to_exclude: List[str] = None,
    verbose: bool = True,
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
    """
    warnings.warn(
        "check_estimator is currently being reworked and does not cover"
        "the whole testing suite. For full coverage, use check_estimator_legacy.",
        UserWarning,
        stacklevel=1,
    )

    _check_estimator_deps(estimator)

    if _is_legacy_estimator(estimator):
        return check_estimator_legacy(
            estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=checks_to_run,
            tests_to_exclude=checks_to_exclude,
            fixtures_to_run=full_checks_to_run,
            fixtures_to_exclude=full_checks_to_exclude,
            verbose=verbose,
        )

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

    passed = 0
    skipped = 0
    failed = 0
    results = {}
    for est, check in checks_generator():
        name = _get_check_estimator_ids(check)
        if isclass(estimator):
            name += f"[{_get_check_estimator_ids(est)}]"

        if checks_to_run is not None and name.split("[")[0] not in checks_to_run:
            continue
        if checks_to_exclude is not None and name.split("[")[0] in checks_to_exclude:
            continue
        if full_checks_to_run is not None and name not in full_checks_to_run:
            continue
        if full_checks_to_exclude is not None and name in full_checks_to_exclude:
            continue

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
