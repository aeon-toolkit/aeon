"""Tests for all early classifiers."""

from functools import partial
from sys import platform

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing.expected_results.expected_classifier_outputs import (
    basic_motions_proba,
    unit_test_proba,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.utils.validation import get_n_cases


def _yield_early_classification_checks(estimator_class, estimator_instances, datatypes):
    """Yield all early classification checks for an aeon early classifier."""
    # only class required
    yield partial(
        check_early_classifier_against_expected_results, estimator_class=estimator_class
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_classifier_output,
                estimator=estimator,
                datatype=datatype,
            )


def check_early_classifier_against_expected_results(estimator_class):
    """Test early classifier against stored results."""
    # we only use the first estimator instance for testing
    classname = estimator_class.__name__

    # We cannot guarantee same results on ARM macOS
    if platform == "darwin":
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
        estimator_instance = estimator_class._create_test_instance(
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
        y_proba, _ = estimator_instance.predict_proba(X_test[indices])

        # assert probabilities are the same
        assert_array_almost_equal(
            y_proba,
            expected_probas,
            decimal=2,
            err_msg=f"Failed to reproduce results for {classname} on {data_name}",
        )


def check_classifier_output(estimator, datatype):
    """Test classifier outputs the correct data types and values.

    Test predict produces a np.array or pd.Series with only values seen in the train
    data, and that predict_proba probability estimates add up to one.
    """
    estimator = _clone_estimator(estimator)

    unique_labels = np.unique(FULL_TEST_DATA_DICT[datatype]["train"][1])

    # run fit and predict
    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred, decisions = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])

    # check predict
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)
    assert np.all(np.isin(np.unique(y_pred), unique_labels))
    assert isinstance(decisions, np.ndarray)
    assert decisions.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)
    assert decisions.dtype == bool

    # predict and update methods should update the state info as an array
    assert isinstance(estimator.get_state_info(), np.ndarray)

    # check predict proba (all classifiers have predict_proba by default)
    y_proba, decisions = estimator.predict_proba(
        FULL_TEST_DATA_DICT[datatype]["test"][0]
    )

    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (
        get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),
        len(unique_labels),
    )
    np.testing.assert_allclose(y_proba.sum(axis=1), 1)
    assert isinstance(decisions, np.ndarray)
    assert decisions.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)
    assert decisions.dtype == bool
