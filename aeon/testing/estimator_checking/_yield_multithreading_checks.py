import inspect
from functools import partial

from aeon.base._base import _clone_estimator
from aeon.testing.testing_config import (
    MULTITHREAD_TESTING,
    NON_STATE_CHANGING_METHODS_ARRAYLIKE,
)
from aeon.testing.utils.deep_equals import deep_equals
from aeon.testing.utils.estimator_checks import _get_tag, _run_estimator_method
from aeon.utils.validation import check_n_jobs


def _yield_multithreading_checks(estimator_class, estimator_instances, datatypes):
    """Yield all multithreading checks for an aeon estimator."""
    can_thread = _get_tag(estimator_class, "capability:multithreading")

    # only class required
    if can_thread:
        yield partial(check_multithreading_param, estimator_class=estimator_class)
    else:
        yield partial(check_no_multithreading_param, estimator_class=estimator_class)

    if can_thread and MULTITHREAD_TESTING:
        # test class instances
        for i, estimator in enumerate(estimator_instances):
            # test all data types
            for datatype in datatypes[i]:
                yield partial(
                    check_estimator_multithreading,
                    estimator=estimator,
                    datatype=datatype,
                )


def check_multithreading_param(estimator_class):
    """Test that estimators that can multithread have a n_jobs parameter."""
    default_params = inspect.signature(estimator_class.__init__).parameters
    n_jobs = default_params.get("n_jobs", None)

    # check that the estimator has a n_jobs parameter
    if n_jobs is None:
        raise ValueError(
            f"{estimator_class} which sets "
            "capability:multithreading=True must have a n_jobs parameter."
        )

    # check that the default value is to use 1 thread
    if n_jobs.default != 1:
        raise ValueError(
            "n_jobs parameter must have a default value of 1, "
            "disabling multithreading by default."
        )

    # test parameters should not change the default value
    params = estimator_class._get_test_params()
    if not isinstance(params, list):
        params = [params]
    for param_set in params:
        assert "n_jobs" not in param_set


def check_no_multithreading_param(estimator_class):
    """Test that estimators that can't multithread have no n_jobs parameter."""
    default_params = inspect.signature(estimator_class.__init__).parameters

    # check that the estimator does not have a n_jobs parameter
    if default_params.get("n_jobs", None) is not None:
        raise ValueError(
            f"{estimator_class} has a n_jobs parameter, but does not set "
            "capability:multithreading=True in its tags."
        )


def check_estimator_multithreading(estimator, datatype):
    """Test that multithreaded estimators store n_jobs_ and produce same results."""
    estimator_name = estimator.__class__.__name__
    st_estimator = _clone_estimator(estimator, random_state=42)
    mt_estimator = _clone_estimator(estimator, random_state=42)

    n_jobs = max(2, check_n_jobs(-2))
    mt_estimator.set_params(n_jobs=n_jobs)

    tags = estimator.get_tags()

    if not tags["fit_is_empty"]:
        # fit and get results for single thread estimator
        _run_estimator_method(st_estimator, "fit", datatype, "train")

        # check _n_jobs attribute is set
        assert hasattr(st_estimator, "_n_jobs"), (
            f"Estimator with default n_jobs {estimator_name} does not store an _n_jobs "
            "attribute. It is recommended to use the "
            "aeon.utils.validation.check_n_jobs function to set _n_jobs and use this "
            "for any multithreading."
        )
        assert st_estimator._n_jobs == 1, (
            f"Estimator with default n_jobs {estimator_name} does not store an _n_jobs "
            f"attribute correctly. Expected 1, got {mt_estimator._n_jobs}."
            f"It is recommended to use the aeon.utils.validation.check_n_jobs function "
            f"to set _n_jobs and use this for any multithreading."
        )

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(st_estimator, method) and callable(getattr(st_estimator, method)):
            output = _run_estimator_method(st_estimator, method, datatype, "test")
            results.append(output)

    if not tags["fit_is_empty"]:
        # fit multithreaded estimator
        _run_estimator_method(mt_estimator, "fit", datatype, "train")

        # check _n_jobs attribute is set
        assert hasattr(mt_estimator, "_n_jobs"), (
            f"Multithreaded estimator {estimator_name} does not store an _n_jobs "
            "attribute. It is recommended to use the "
            "aeon.utils.validation.check_n_jobs function to set _n_jobs and use this "
            "for any multithreading."
        )
        assert mt_estimator._n_jobs == n_jobs, (
            f"Multithreaded estimator {estimator_name} does not store an _n_jobs "
            f"attribute correctly. Expected {n_jobs}, got {mt_estimator._n_jobs}."
            f"It is recommended to use the aeon.utils.validation.check_n_jobs function "
            f"to set _n_jobs and use this for any multithreading."
        )

    # compare results from single and multithreaded estimators
    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(mt_estimator, method) and callable(getattr(mt_estimator, method)):
            output = _run_estimator_method(mt_estimator, method, datatype, "test")

            if not tags["non_deterministic"]:
                assert deep_equals(output, results[i]), (
                    f"Running {method} after fit with test parameters gives different "
                    f"results when multithreading."
                )
            i += 1
