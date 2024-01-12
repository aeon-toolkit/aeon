"""Unit tests for classifier/regressor input output."""

__author__ = ["mloning", "TonyBagnall", "fkiraly"]

import inspect

import numpy as np
from sklearn.utils._testing import set_random_state

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing import (
    BaseFixtureGenerator,
    QuickTester,
    basic_motions_proba,
    unit_test_proba,
)
from aeon.utils._testing.estimator_checks import _assert_array_almost_equal
from aeon.utils._testing.scenarios_classification import ClassifierFitPredict
from aeon.utils.validation.collection import get_n_cases


class ClassifierFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for classifier tests.

    Fixtures parameterized
    ----------------------
    estimator_class: estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
    estimator_instance: instance of estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method
    scenario: instance of TestScenario
        ranges over all scenarios returned by retrieve_scenarios
    """

    # note: this should be separate from TestAllClassifiers
    #   additional fixtures, parameters, etc should be added here
    #   Classifiers should contain the tests only

    estimator_type_filter = "classifier"


class TestAllClassifiers(ClassifierFixtureGenerator, QuickTester):
    """Module level tests for all aeon classifiers."""

    def test_classifier_output(self, estimator_instance, scenario):
        """Test classifier outputs the correct data types and values.

        Test predict produces a np.array or pd.Series with only values seen in the train
        data, and that predict_proba probability estimates add up to one.
        """
        n_classes = scenario.get_tag("n_classes")
        X = scenario.args["predict"]["X"]
        y = scenario.args["fit"]["y"]
        n_cases = get_n_cases(X)

        # run fit and predict
        y_pred = scenario.run(estimator_instance, method_sequence=["fit", "predict"])

        # check predict
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (n_cases,)
        assert np.all(np.isin(np.unique(y_pred), np.unique(y)))

        # check predict proba (all classifiers have predict_proba by default)
        y_proba = scenario.run(estimator_instance, method_sequence=["predict_proba"])
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (n_cases, n_classes)
        np.testing.assert_almost_equal(y_proba.sum(axis=1), 1, decimal=4)

    def test_classifier_against_expected_results(self, estimator_class):
        """Test classifier against stored results."""
        # we only use the first estimator instance for testing
        classname = estimator_class.__name__

        # the test currently fails when numba is disabled. See issue #622
        import os

        if classname == "HIVECOTEV2" and os.environ.get("NUMBA_DISABLE_JIT") == "1":
            return None

        for data_name, data_dict, data_loader, data_seed in [
            ["UnitTest", unit_test_proba, load_unit_test, 0],
            ["BasicMotions", basic_motions_proba, load_basic_motions, 4],
        ]:
            # retrieve expected predict_proba output, and skip test if not available
            if classname in data_dict.keys():
                expected_probas = data_dict[classname]
            else:
                # skip test if no expected probas are registered
                continue

            # we only use the first estimator instance for testing
            estimator_instance = estimator_class.create_test_instance(
                parameter_set="results_comparison"
            )
            # set random seed if possible
            set_random_state(estimator_instance, 0)

            # load test data
            X_train, y_train = data_loader(split="train")
            X_test, _ = data_loader(split="test")
            indices = np.random.RandomState(data_seed).choice(
                len(y_train), 10, replace=False
            )

            # train classifier and predict probas
            estimator_instance.fit(X_train[indices], y_train[indices])
            y_proba = estimator_instance.predict_proba(X_test[indices])

            # assert probabilities are the same
            _assert_array_almost_equal(
                y_proba,
                expected_probas,
                decimal=2,
                err_msg=f"Failed to reproduce results for {classname} on {data_name}",
            )

    def test_contracted_classifier(self, estimator_class):
        """Test classifiers that can be contracted."""
        if estimator_class.get_class_tag(tag_name="capability:contractable") is True:
            # if we have a train_estimate parameter set use it, else use default
            estimator_instance = estimator_class.create_test_instance(
                parameter_set="contracting"
            )

            # The "capability:contractable" has not been fully implemented yet.
            # Most uses currently have a time_limit_in_minutes parameter, but we won't
            # fail those that don't.
            default_params = inspect.signature(estimator_class.__init__).parameters
            if default_params.get(
                "time_limit_in_minutes", None
            ) is not None and default_params.get(
                "time_limit_in_minutes", None
            ).default not in (
                0,
                -1,
                None,
            ):
                return None

            # too short of a contract time can lead to test failures
            if vars(estimator_instance).get("time_limit_in_minutes", None) < 5:
                raise ValueError(
                    "Test parameters for test_contracted_classifier must set "
                    "time_limit_in_minutes to 5 or more."
                )

            scenario = ClassifierFitPredict()

            X_new = scenario.args["predict"]["X"]
            y_train = scenario.args["fit"]["y"]
            X_new_instances = get_n_cases(X_new)

            # run fit and predict
            y_pred = scenario.run(
                estimator_instance, method_sequence=["fit", "predict"]
            )

            # check predict
            assert isinstance(y_pred, np.ndarray)
            assert y_pred.shape == (X_new_instances,)
            assert np.all(np.isin(np.unique(y_pred), np.unique(y_train)))
        else:
            # skip test if it can't contract
            return None

    def test_classifier_train_estimate(self, estimator_class):
        """Test classifiers that can produce train set probability estimates."""
        if estimator_class.get_class_tag(tag_name="capability:train_estimate") is True:
            # if we have a train_estimate parameter set use it, else use default
            estimator_instance = estimator_class.create_test_instance(
                parameter_set="train_estimate"
            )

            # The "capability:train_estimate" has not been fully implemented yet.
            # Most uses currently have the below method, but we won't fail those that
            # don't.
            if not hasattr(estimator_instance, "_get_train_probs"):
                return None

            # fit classifier
            scenario = ClassifierFitPredict()
            scenario.run(estimator_instance, method_sequence=["fit"])

            n_classes = scenario.get_tag("n_classes")
            X_train = scenario.args["fit"]["X"]
            y_train = scenario.args["fit"]["y"]
            X_train_len = get_n_cases(X_train)

            # check the probabilities are valid
            train_proba = estimator_instance._get_train_probs(X_train, y_train)
            assert isinstance(train_proba, np.ndarray)
            assert train_proba.shape == (X_train_len, n_classes)
            np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)
        else:
            # skip test if it can't produce an estimate
            return None

    def test_classifier_tags_consistent(self, estimator_class):
        """Test the tag X_inner_type is consistent with capability:unequal_length."""
        valid_types = {"np-list", "df-list", "pd-multivariate", "nested_univ"}
        unequal = estimator_class.get_class_tag("capability:unequal_length")
        if unequal:  # one of X_inner_types must be capable of storing unequal length
            internal_types = estimator_class.get_class_tag("X_inner_type")
            if isinstance(internal_types, str):
                assert internal_types in valid_types
            else:  # must be a list
                assert bool(set(internal_types) & valid_types)
        # Test can actually fit/predict with multivariate if tag is set
        multivariate = estimator_class.get_class_tag("capability:multivariate")
        if multivariate:
            X = np.random.random((10, 2, 20))
            y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
            inst = estimator_class.create_test_instance(parameter_set="default")
            inst.fit(X, y)
            inst.predict(X)
            inst.predict_proba(X)

    def test_does_not_override_final_methods(self, estimator_class):
        """Test does not override final methods."""
        if "fit" in estimator_class.__dict__:
            raise ValueError(f"Classifier {estimator_class} overrides the method fit")
        if "predict" in estimator_class.__dict__:
            raise ValueError(
                f"Classifier {estimator_class} overrides the method " f"predict"
            )
        if "predict_proba" in estimator_class.__dict__:
            raise ValueError(
                f"Classifier {estimator_class} overrides the method " f"predict_proba"
            )
