import inspect
from functools import partial

from numpy.testing import assert_array_almost_equal

from aeon.base._base import _clone_estimator
from aeon.testing.testing_config import (
    MULTITHREAD_TESTING,
    NON_STATE_CHANGING_METHODS_ARRAYLIKE,
)
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
    """Test that estimators that cant multithread have no n_jobs parameter."""
    default_params = inspect.signature(estimator_class.__init__).parameters

    # check that the estimator does not have a n_jobs parameter
    if default_params.get("n_jobs", None) is not None:
        raise ValueError(
            f"{estimator_class} has a n_jobs parameter, but does not set "
            "capability:multithreading=True in its tags."
        )


def check_estimator_multithreading(estimator, datatype):
    """Test that multithreaded estimators store n_jobs_ and produce same results."""
    st_estimator = _clone_estimator(estimator, random_state=42)
    mt_estimator = _clone_estimator(estimator, random_state=42)
    n_jobs = max(2, check_n_jobs(-2))
    mt_estimator.set_params(n_jobs=n_jobs)

    # fit and get results for single thread estimator
    _run_estimator_method(st_estimator, "fit", datatype, "train")

    results = []
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(st_estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(st_estimator, method, datatype, "test")
            results.append(output)

    # fit multithreaded estimator
    _run_estimator_method(mt_estimator, "fit", datatype, "train")

    # check n_jobs_ attribute is set
    assert mt_estimator.n_jobs_ == n_jobs, (
        f"Multithreaded estimator {mt_estimator} does not store n_jobs_ "
        f"attribute correctly. Expected {n_jobs}, got {mt_estimator.n_jobs_}."
        f"It is recommended to use the check_n_jobs function to set n_jobs_ and use"
        f"this for any multithreading."
    )

    # compare results from single and multithreaded estimators
    i = 0
    for method in NON_STATE_CHANGING_METHODS_ARRAYLIKE:
        if hasattr(estimator, method) and callable(getattr(estimator, method)):
            output = _run_estimator_method(estimator, method, datatype, "test")

            assert_array_almost_equal(
                output,
                results[i],
                err_msg=f"Running {method} after fit twice with test "
                f"parameters gives different results.",
            )
            i += 1
