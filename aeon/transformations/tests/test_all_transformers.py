"""Unit tests common to all transformers."""

__author__ = ["mloning", "fkiraly", "MatthewMiddlehurst"]
__all__ = []

import numpy as np
import pandas as pd
from sklearn.testing.utils import set_random_state

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing.expected_results.expected_transform_outputs import (
    basic_motions_result,
    unit_test_result,
)
from aeon.testing.test_all_estimators import BaseFixtureGenerator, QuickTester
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal


class TransformerFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for transformer tests.

    Fixtures parameterized
    ----------------------
    estimator_class: estimator inheriting from BaseObject
        ranges over all estimator classes not excluded by EXCLUDED_TESTS
    estimator_instance: instance of estimator inheriting from BaseObject
        ranges over all estimator classes not excluded by EXCLUDED_TESTS
        instances are generated by create_test_instance class method
    scenario: instance of TestScenario
        ranges over all scenarios returned by retrieve_scenarios
    """

    # note: this should be separate from TestAllTransformers
    #   additional fixtures, parameters, etc should be added here
    #   TestAllTransformers should contain the tests only

    estimator_type_filter = "transformer"


class TestAllTransformers(TransformerFixtureGenerator, QuickTester):
    """Module level tests for all aeon transformers."""

    def test_capability_inverse_tag_is_correct(self, estimator_instance):
        """Test that the capability:inverse_transform tag is set correctly."""
        capability_tag = estimator_instance.get_tag("capability:inverse_transform")
        skip_tag = estimator_instance.get_tag("skip-inverse-transform")
        if capability_tag and not skip_tag:
            assert estimator_instance._has_implementation_of("_inverse_transform")

    def test_remember_data_tag_is_correct(self, estimator_instance):
        """Test that the remember_data tag is set correctly."""
        fit_empty_tag = estimator_instance.get_tag("fit_is_empty", True)
        remember_data_tag = estimator_instance.get_tag("remember_data", False)
        msg = (
            'if the "remember_data" tag is set to True, then the "fit_is_empty" tag '
            "must be set to False, even if _fit is not implemented or empty. "
            "This is due to boilerplate that write to self.X in fit. "
            f"Please check these two tags in {type(estimator_instance)}."
        )
        if fit_empty_tag and remember_data_tag:
            raise AssertionError(msg)

    def test_transform_inverse_transform_equivalent(self, estimator_instance, scenario):
        """Test that inverse_transform is indeed inverse to transform."""
        # skip this test if the estimator does not have inverse_transform
        if not estimator_instance.get_class_tag("capability:inverse_transform", False):
            return None

        # skip this test if the estimator skips inverse_transform
        if estimator_instance.get_tag("skip-inverse-transform", False):
            return None

        X = scenario.args["transform"]["X"]
        Xt = scenario.run(estimator_instance, method_sequence=["fit", "transform"])
        Xit = estimator_instance.inverse_transform(Xt)
        if estimator_instance.get_tag("transform-returns-same-time-index"):
            _assert_array_almost_equal(X, Xit)
        elif isinstance(X, pd.DataFrame):
            _assert_array_almost_equal(X.loc[Xit.index], Xit)

    def test_transformer_against_expected_results(self, estimator_class):
        """Test transformer against stored results."""
        # we only use the first estimator instance for testing
        classname = estimator_class.__name__

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

            # fir transformer and transform
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
