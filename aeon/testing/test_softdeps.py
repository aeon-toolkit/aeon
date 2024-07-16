"""Tests that soft dependencies are handled correctly.

aeon supports a number of soft dependencies which are necessary for using
a certain module or estimator but otherwise not necessary.
"""

__maintainer__ = []

import pkgutil
import re
from importlib import import_module

import pytest

import aeon
from aeon.registry import all_estimators
from aeon.testing.test_config import EXCLUDE_ESTIMATORS, PR_TESTING
from aeon.testing.utils.scenarios_getter import retrieve_scenarios
from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)

# collect all modules
modules = pkgutil.walk_packages(aeon.__path__, aeon.__name__ + ".")
modules = [x[1] for x in modules]

if PR_TESTING:  # pragma: no cover
    # exclude test modules
    modules = [x for x in modules if not any(part == "tests" for part in x.split("."))]


def test_module_crawl():
    """Test that we are crawling modules correctly."""
    assert "aeon.classification" in modules
    assert "aeon.classification.shapelet_based" in modules
    assert "aeon.classification.base" in modules
    assert "aeon.forecasting" in modules


@pytest.mark.parametrize("module", modules)
def test_module_soft_deps(module):
    """Test soft dependency imports in aeon modules.

    Imports all modules and catch exceptions due to missing dependencies.
    """
    try:
        import_module(module)
    except ModuleNotFoundError as e:  # pragma: no cover
        dependency = "unknown"
        match = re.search(r"\'(.+?)\'", str(e))
        if match:
            dependency = match.group(1)

        raise ModuleNotFoundError(
            f"The module: {module} should not require any soft dependencies, "
            f"but tried importing: '{dependency}'. Make sure soft dependencies are "
            f"properly isolated."
        ) from e


# TODO test revamp: this can be part a greater check of all estimators probably, dont
# need to discover all estimators again here


def _has_soft_dep(est):
    """Return whether an estimator has soft dependencies."""
    softdep = est.get_class_tag("python_dependencies", None)
    return softdep is not None


def _coerce_list_of_str(obj):
    """Coerce obj to list of str."""
    if obj is None:
        return []
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, list):
        return obj


def _get_soft_deps(est):
    """Return soft dependencies of an estimator, as list of str."""
    softdeps = est.get_class_tag("python_dependencies", None)
    softdeps = _coerce_list_of_str(softdeps)
    if softdeps is None:
        raise RuntimeError(
            'error, "python_dependencies" tag must be None, str or list of str,'
            f" but {est.__name__} has {softdeps}"
        )
    else:
        return softdeps


def _is_in_env(modules):
    """Return whether all modules in list of str modules are installed in env."""
    modules = _coerce_list_of_str(modules)
    try:
        for module in modules:
            _check_soft_dependencies(module, severity="error")
        return True
    except ModuleNotFoundError:
        return False


# all estimators - exclude estimators on the global exclusion list
all_ests = all_estimators(return_names=False, exclude_estimators=EXCLUDE_ESTIMATORS)


# estimators that should fail to construct because of python version
est_python_incompatible = [
    est for est in all_ests if not _check_python_version(est, severity="none")
]

# estimators that have soft dependencies
est_with_soft_dep = [est for est in all_ests if _has_soft_dep(est)]
# estimators that have soft dependencies and are python compatible
est_pyok_with_soft_dep = [
    est for est in est_with_soft_dep if _check_python_version(est, severity="none")
]

# estimators that have no soft dependenies
est_without_soft_dep = [est for est in all_ests if not _has_soft_dep(est)]
# estimators that have soft dependencies and are python compatible
est_pyok_without_soft_dep = [
    est for est in est_without_soft_dep if _check_python_version(est, severity="none")
]

# all estimators are now a disjoint union of the three sets:
# est_python_incompatible - python incompatible, should raise python error
# est_pyok_without_soft_dep - python compatible, has no soft dependency
# est_pyok_with_soft_dep - python compatible, has soft dependency


@pytest.mark.parametrize("estimator", est_python_incompatible)
def test_python_error(estimator):
    """Test that estimators raise error if python version is wrong."""
    try:
        estimator.create_test_instance()
    except ModuleNotFoundError as e:
        error_msg = str(e)

        # Check if appropriate exception with useful error message is raised as
        # defined in the `_check_python` function
        expected_error_msg = "requires python version to be"
        if expected_error_msg not in error_msg:
            pyspec = estimator.get_class_tag("python_version", None)
            raise RuntimeError(
                f"Estimator {estimator.__name__} has python version bound "
                f"{pyspec} according to tags, but does not raise an appropriate "
                f"error message on __init__ for incompatible python environments. "
                f"Likely reason is that __init__ does not call super(cls).__init__."
            ) from e


@pytest.mark.parametrize("estimator", est_pyok_with_soft_dep)
def test_softdep_error(estimator):
    """Test that estimators raise error if required soft dependencies are missing."""
    softdeps = _get_soft_deps(estimator)
    if not _is_in_env(softdeps):
        try:
            estimator.create_test_instance()
        except ModuleNotFoundError as e:
            error_msg = str(e)

            # Check if appropriate exception with useful error message is raised as
            # defined in the `_check_soft_dependencies` function
            expected_error_msg = (
                "is a soft dependency and not included in the base aeon installation"
            )
            # message is different for deep learning deps tensorflow, tensorflow-proba
            error_msg_alt = "required for deep learning"
            if "incompatible version" in error_msg:
                pass
            elif expected_error_msg not in error_msg and error_msg_alt not in error_msg:
                raise RuntimeError(
                    f"Estimator {estimator.__name__} requires soft dependencies "
                    f"{softdeps} according to tags, but does not raise an appropriate "
                    f"error message on __init__, when the soft dependency is missing. "
                    f"Likely reason is that __init__ does not call super(cls).__init__,"
                    f" or imports super(cls).__init__ only after an attempted import."
                ) from e


@pytest.mark.parametrize("estimator", est_pyok_with_soft_dep)
def test_est_construct_if_softdep_available(estimator):
    """Test that estimators construct if required soft dependencies are there."""
    softdeps = _get_soft_deps(estimator)
    if _is_in_env(softdeps):
        try:
            estimator.create_test_instance()
        except ModuleNotFoundError as e:
            error_msg = str(e)
            raise RuntimeError(
                f"Estimator {estimator.__name__} requires soft dependencies "
                f"{softdeps} according to tags, but raises ModuleNotFoundError "
                f"on __init__ when those dependencies are in the environment. "
                f" Likely cause is additionally needed soft dependencies, "
                f"these should be added "
                f'to the "python_dependencies" tag. Exception text: {error_msg}'
            ) from e


@pytest.mark.parametrize("estimator", est_pyok_without_soft_dep)
def test_est_construct_without_modulenotfound(estimator):
    """Test that estimators that do not require soft dependencies construct properly."""
    try:
        estimator.create_test_instance()
    except ModuleNotFoundError as e:
        error_msg = str(e)
        raise RuntimeError(
            f"Estimator {estimator.__name__} does not require soft dependencies "
            f"according to tags, but raises ModuleNotFoundError "
            f"on __init__ with test parameters. Any required soft dependencies should "
            f'be added to the "python_dependencies" tag, and python version bounds '
            f'should be added to the "python_version" tag. Exception text: {error_msg}'
        ) from e


@pytest.mark.parametrize("estimator", est_pyok_without_soft_dep)
def test_est_fit_without_modulenotfound(estimator):
    """Test that estimators that do not require soft dependencies fit properly."""
    try:
        scenarios = retrieve_scenarios(estimator)
        if len(scenarios) == 0:
            return None
        else:
            scenario = scenarios[0]
        estimator_instance = estimator.create_test_instance()
        scenario.run(estimator_instance, method_sequence=["fit"])
    except ModuleNotFoundError as e:
        error_msg = str(e)
        raise RuntimeError(
            f"Estimator {estimator.__name__} does not require soft dependencies "
            f"according to tags, but raises ModuleNotFoundError "
            f"on fit. Any required soft dependencies should be added "
            f'to the "python_dependencies" tag, and python version bounds should be'
            f' added to the "python_version" tag. Exception text: {error_msg}'
        ) from e
