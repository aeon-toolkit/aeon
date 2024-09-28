"""Tests for all transformers."""

from functools import partial
from sys import platform

import numpy as np
import pandas as pd
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing.expected_results.expected_transform_outputs import (
    basic_motions_result,
    unit_test_result,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import (
    _assert_array_almost_equal,
    _run_estimator_method,
)


def _yield_transformation_checks(estimator_class, estimator_instances, datatypes):
    """Yield all transformation checks for an aeon transformer."""
    # only class required
    yield partial(
        check_transformer_against_expected_results, estimator_class=estimator_class
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # no data needed
        yield partial(
            check_capability_inverse_tag_is_correct,
            estimator=estimator,
        )
        yield partial(
            check_remember_data_tag_is_correct,
            estimator=estimator,
        )

        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_transform_inverse_transform_equivalent,
                estimator=estimator,
                datatype=datatype,
            )


def check_transformer_against_expected_results(estimator_class):
    """Test transformer against stored results."""
    # we only use the first estimator instance for testing
    classname = estimator_class.__name__

    # We cannot guarantee same results on ARM macOS
    if platform == "darwin":
        return None

    for data_name, data_dict, data_loader, data_seed in [
        ["UnitTest", unit_test_result, load_unit_test, 0],
        ["BasicMotions", basic_motions_result, load_basic_motions, 4],
    ]:
        # retrieve expected transform output, and skip test if not available
        if classname in data_dict.keys():
            expected_results = data_dict[classname]
        else:
            # skip test if no expected results are registered
            continue

        # we only use the first estimator instance for testing
        estimator_instance = estimator_class.create_test_instance(
            parameter_set="results_comparison"
        )
        # set random seed if possible
        set_random_state(estimator_instance, 0)

        # load test data
        X_train, y_train = data_loader(split="train")
        indices = np.random.RandomState(data_seed).choice(
            len(y_train), 5, replace=False
        )

        # fit transformer and transform
        results = np.nan_to_num(
            estimator_instance.fit_transform(X_train[indices], y_train[indices]),
            False,
            0,
            0,
            0,
        )

        # assert results are the same
        _assert_array_almost_equal(
            results,
            expected_results,
            decimal=2,
            err_msg=f"Failed to reproduce results for {classname} on {data_name}",
        )


def check_capability_inverse_tag_is_correct(estimator):
    """Test that the capability:inverse_transform tag is set correctly."""
    capability_tag = estimator.get_tag("capability:inverse_transform")
    skip_tag = estimator.get_tag("skip-inverse-transform")
    if capability_tag and not skip_tag:
        assert estimator._has_implementation_of("_inverse_transform")


def check_remember_data_tag_is_correct(estimator):
    """Test that the remember_data tag is set correctly."""
    fit_empty_tag = estimator.get_tag("fit_is_empty", True)
    remember_data_tag = estimator.get_tag("remember_data", False)
    msg = (
        'if the "remember_data" tag is set to True, then the "fit_is_empty" tag '
        "must be set to False, even if _fit is not implemented or empty. "
        "This is due to boilerplate that write to self.X in fit. "
        f"Please check these two tags in {type(estimator)}."
    )
    if fit_empty_tag and remember_data_tag:
        raise AssertionError(msg)


def check_transform_inverse_transform_equivalent(estimator, datatype):
    """Test that inverse_transform is indeed inverse to transform."""
    # skip this test if the estimator does not have inverse_transform
    if not estimator.get_class_tag("capability:inverse_transform", False):
        return None

    # skip this test if the estimator skips inverse_transform
    if estimator.get_tag("skip-inverse-transform", False):
        return None

    estimator = _clone_estimator(estimator)

    X = FULL_TEST_DATA_DICT[datatype]["train"][0]

    _run_estimator_method(estimator, "fit", datatype, "train")
    Xt = _run_estimator_method(estimator, "transform", datatype, "train")

    Xit = estimator.inverse_transform(Xt)
    if estimator.get_tag("transform-returns-same-time-index"):
        _assert_array_almost_equal(X, Xit)
    elif isinstance(X, pd.DataFrame):
        _assert_array_almost_equal(X.loc[Xit.index], Xit)
