"""Tests for estimator soft dependencies.

Only tests with 'softdep' in the name will be run by `check_estimator` if any
required package or version is missing. Other tests will be automatically skipped.
"""

from functools import partial

from aeon.utils.validation._dependencies import (
    _check_python_version,
    _check_soft_dependencies,
)


def _yield_soft_dependency_checks(estimator_class, estimator_instances, datatypes):
    """Yield all soft dependency checks for an aeon estimator."""
    # only class required
    yield partial(check_python_version_softdep, estimator_class=estimator_class)
    yield partial(check_python_dependency_softdep, estimator_class=estimator_class)


def check_python_version_softdep(estimator_class):
    """Test that estimators raise error if python version is wrong."""
    import pytest

    # if dependencies are incompatible skip
    softdeps = estimator_class.get_class_tag("python_dependencies", None)
    if softdeps is not None and not _check_soft_dependencies(softdeps, severity="none"):
        return

    # should be compatible with python version and able to construct
    if _check_python_version(estimator_class, severity="none"):
        estimator_class._create_test_instance()
    # should raise a specific error if python version is incompatible
    else:
        pyspec = estimator_class.get_class_tag("python_version", None)
        with pytest.raises(ModuleNotFoundError) as ex_info:
            estimator_class._create_test_instance()
        assert "requires python version to be" in str(ex_info.value), (
            f"Estimator {estimator_class.__name__} has python version bound "
            f"{pyspec} according to tags, but does not raise an appropriate "
            f"error message on __init__ for incompatible python environments. "
            f"Likely reason is that __init__ does not call super(cls).__init__."
        )


def check_python_dependency_softdep(estimator_class):
    """Test that estimators raise error if required soft dependencies are missing."""
    import pytest

    # if python version is incompatible skip
    if not _check_python_version(estimator_class, severity="none"):
        return

    softdeps = estimator_class.get_class_tag("python_dependencies", None)

    # should be compatible with installed dependencies and able to construct
    if softdeps is None or _check_soft_dependencies(softdeps, severity="none"):
        estimator_class._create_test_instance()
    # should raise a specific error if any soft dependencies are missing
    else:
        with pytest.raises(ModuleNotFoundError) as ex_info:
            estimator_class._create_test_instance()
        assert (
            "is a soft dependency and not included in the base aeon installation"
            in str(ex_info.value)
        ), (
            f"Estimator {estimator_class.__name__} requires soft dependencies "
            f"{softdeps} according to tags, but does not raise an appropriate "
            f"error message on __init__, when a soft dependency is missing. "
            f"Likely reason is that __init__ does not call super(cls).__init__, "
            f"or imports super(cls).__init__ only after an attempted import."
        )
