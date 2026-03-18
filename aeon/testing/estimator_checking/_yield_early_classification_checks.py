"""Tests for all early classifiers."""

import sys
from copy import deepcopy
from functools import partial

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.ensemble._base import _set_random_states

from aeon.base._base import _clone_estimator
from aeon.testing.expected_results._write_estimator_results import (
    X_bm_test,
    X_bm_train,
    X_ut_test,
    X_ut_train,
    y_bm_train,
    y_ut_train,
)
from aeon.testing.expected_results.expected_early_classifier_results import (
    multivariate_expected_results,
    univariate_expected_results,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.utils.validation.collection import get_n_cases


def _yield_early_classification_checks(estimator_class, estimator_instances, datatypes):
    """Yield all early classification checks for an aeon early classifier."""
    # only class required
    yield partial(
        check_early_classifier_against_expected_results,
        estimator_class=estimator_class,
        data_name="UnitTest",
        X_train=X_ut_train,
        y_train=y_ut_train,
        X_test=X_ut_test,
        results_dict=univariate_expected_results,
    )
    yield partial(
        check_early_classifier_against_expected_results,
        estimator_class=estimator_class,
        data_name="BasicMotions",
        X_train=X_bm_train,
        y_train=y_bm_train,
        X_test=X_bm_test,
        results_dict=multivariate_expected_results,
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_early_classifier_output,
                estimator=estimator,
                datatype=datatype,
            )


def check_early_classifier_against_expected_results(
    estimator_class,
    data_name,
    X_train,
    y_train,
    X_test,
    results_dict,
):
    """Test early classifier against stored results."""
    # retrieve expected predict_proba output, and skip test if not available
    if sys.platform != "linux":
        # we cannot guarantee same results on ARM macOS
        return "Comparison against expected results is only available on Linux."
    elif estimator_class.__name__ in results_dict.keys():
        expected_probas = results_dict[estimator_class.__name__]
    else:
        # skip test if no expected probas are registered
        return f"No stored results for {estimator_class.__name__} on {data_name}"

    # we only use the first estimator instance for testing
    estimator_instance = estimator_class._create_test_instance(
        parameter_set="results_comparison", return_first=True
    )
    # set random seed if possible
    _set_random_states(estimator_instance, 42)

    # train early classifier and predict probas
    estimator_instance.fit(deepcopy(X_train), deepcopy(y_train))
    y_proba, decisions = estimator_instance.predict_proba(deepcopy(X_test))

    # assert probabilities are the same
    assert_array_almost_equal(
        y_proba,
        expected_probas,
        decimal=2,
        err_msg=(
            f"Failed to reproduce results for {estimator_class.__name__} "
            f"on {data_name}"
        ),
    )


def check_early_classifier_output(estimator, datatype):
    """Test early classifier outputs the correct data types and values.

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
