"""Tests for all regressors."""

import inspect
import os
import sys
import tempfile
import time
from functools import partial

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.regression.deep_learning import BaseDeepRegressor
from aeon.testing.expected_results.expected_regressor_outputs import (
    cardano_sentiment_preds,
    covid_3month_preds,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import _assert_predict_labels, _get_tag
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES


def _yield_regression_checks(estimator_class, estimator_instances, datatypes):
    """Yield all regression checks for an aeon regressor."""
    # only class required
    if sys.platform == "linux":  # We cannot guarantee same results on ARM macOS
        # Compare against results for both Covid3Month and CardanoSentiment if available
        yield partial(
            check_regressor_against_expected_results,
            estimator_class=estimator_class,
            data_name="Covid3Month",
            data_loader=load_covid_3month,
            results_dict=covid_3month_preds,
            resample_seed=0,
        )
        yield partial(
            check_regressor_against_expected_results,
            estimator_class=estimator_class,
            data_name="CardanoSentiment",
            data_loader=load_cardano_sentiment,
            results_dict=cardano_sentiment_preds,
            resample_seed=0,
        )
    yield partial(check_regressor_overrides_and_tags, estimator_class=estimator_class)

    # data type irrelevant
    if _get_tag(estimator_class, "capability:contractable", raise_error=True):
        yield partial(
            check_contracted_regressor,
            estimator_class=estimator_class,
            datatype=datatypes[0][0],
        )

    if issubclass(estimator_class, BaseDeepRegressor):
        yield partial(
            check_regressor_saving_loading_deep_learning,
            estimator_class=estimator_class,
            datatype=datatypes[0][0],
        )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # data type irrelevant
        if _get_tag(estimator_class, "capability:train_estimate", raise_error=True):
            yield partial(
                check_regressor_train_estimate,
                estimator=estimator,
                datatype=datatypes[0][0],
            )

        if isinstance(estimator, BaseDeepRegressor):
            yield partial(
                check_regressor_random_state_deep_learning,
                estimator=estimator,
                datatype=datatypes[i][0],
            )

        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_regressor_output, estimator=estimator, datatype=datatype
            )


def check_regressor_against_expected_results(
    estimator_class, data_name, data_loader, results_dict, resample_seed
):
    """Test regressor against stored results."""
    # retrieve expected predict output, and skip test if not available
    if estimator_class.__name__ in results_dict.keys():
        expected_preds = results_dict[estimator_class.__name__]
    else:
        # skip test if no expected preds are registered
        return f"No stored results for {estimator_class.__name__} on {data_name}"

    # we only use the first estimator instance for testing
    estimator_instance = estimator_class._create_test_instance(
        parameter_set="results_comparison", return_first=True
    )
    # set random seed if possible
    set_random_state(estimator_instance, 0)

    # load test data
    X_train, y_train = data_loader(split="train")
    X_test, y_test = data_loader(split="test")
    # resample test data
    indices_train = np.random.RandomState(resample_seed).choice(
        len(y_train), 10, replace=False
    )
    indices_test = np.random.RandomState(resample_seed).choice(
        len(y_test), 10, replace=False
    )

    # train regressor and predict
    estimator_instance.fit(X_train[indices_train], y_train[indices_train])
    y_pred = estimator_instance.predict(X_test[indices_test])

    # assert predictions are the same
    assert_array_almost_equal(
        y_pred,
        expected_preds,
        decimal=2,
        err_msg=(
            f"Failed to reproduce results for {estimator_class.__name__} "
            f"on {data_name}"
        ),
    )


def check_regressor_overrides_and_tags(estimator_class):
    """Test compliance with the regressor base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    final_methods = [
        "fit",
        "predict",
        "fit_predict",
    ]
    for method in final_methods:
        if method in estimator_class.__dict__:
            raise ValueError(
                f"Regressor {estimator_class} overrides the method {method}. "
                f"Override _{method} instead."
            )

    # Test valid tag for X_inner_type
    X_inner_type = estimator_class.get_class_tag(tag_name="X_inner_type")
    if isinstance(X_inner_type, str):
        assert X_inner_type in COLLECTIONS_DATA_TYPES
    else:  # must be a list
        assert all([t in COLLECTIONS_DATA_TYPES for t in X_inner_type])

    # one of X_inner_types must be capable of storing unequal length
    if estimator_class.get_class_tag("capability:unequal_length"):
        valid_unequal_types = ["np-list", "df-list", "pd-multiindex"]
        if isinstance(X_inner_type, str):
            assert X_inner_type in valid_unequal_types
        else:  # must be a list
            assert any([t in valid_unequal_types for t in X_inner_type])

    valid_algorithm_types = [
        "distance",
        "deeplearning",
        "convolution",
        "dictionary",
        "interval",
        "feature",
        "hybrid",
        "shapelet",
    ]
    algorithm_type = estimator_class.get_class_tag("algorithm_type")
    if algorithm_type is not None:
        assert algorithm_type in valid_algorithm_types, (
            f"Estimator {estimator_class.__name__} has an invalid 'algorithm_type' "
            f"tag: '{algorithm_type}'. Valid types are {valid_algorithm_types}."
        )


def check_contracted_regressor(estimator_class, datatype):
    """Test regressors that can be contracted."""
    estimator_instance = estimator_class._create_test_instance(
        parameter_set="contracting"
    )
    default_params = inspect.signature(estimator_class.__init__).parameters

    # check that the regressor has a time_limit_in_minutes parameter
    if default_params.get("time_limit_in_minutes", None) is None:
        raise ValueError(
            f"Regressor {estimator_class} which sets "
            "capability:contractable=True must have a time_limit_in_minutes "
            "parameter."
        )

    # check that the default value is to turn off contracting
    if default_params.get("time_limit_in_minutes", None).default not in (
        0,
        -1,
        None,
    ):
        raise ValueError(
            "time_limit_in_minutes parameter must have a default value of 0, "
            "-1 or None, disabling contracting by default."
        )

    # too short of a contract time can lead to test failures
    if vars(estimator_instance).get("time_limit_in_minutes", 0) < 0.5:
        raise ValueError(
            "Test parameters for test_contracted_regressor must set "
            "time_limit_in_minutes to 0.5 or more. It is recommended to make "
            "this larger and add an alternative stopping mechanism "
            "(i.e. max ensemble members)."
        )

    # run fit and predict
    estimator_instance.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred = estimator_instance.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])

    # check predict
    _assert_predict_labels(y_pred, datatype)


def check_regressor_saving_loading_deep_learning(estimator_class, datatype):
    """Test deep regressor saving."""
    with tempfile.TemporaryDirectory() as tmp:
        if tmp[-1] != "/":
            tmp = tmp + "/"

        curr_time = str(time.time_ns())
        last_file_name = curr_time + "last"
        best_file_name = curr_time + "best"
        init_file_name = curr_time + "init"

        deep_rgs_train = estimator_class(
            n_epochs=2,
            save_best_model=True,
            save_last_model=True,
            save_init_model=True,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            init_file_name=init_file_name,
            file_path=tmp,
        )
        deep_rgs_train.fit(
            FULL_TEST_DATA_DICT[datatype]["train"][0],
            FULL_TEST_DATA_DICT[datatype]["train"][1],
        )

        deep_rgs_best = estimator_class()
        deep_rgs_best.load_model(
            model_path=os.path.join(tmp, best_file_name + ".keras"),
        )
        ypred_best = deep_rgs_best.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred_best, datatype)

        deep_rgs_last = estimator_class()
        deep_rgs_last.load_model(
            model_path=os.path.join(tmp, last_file_name + ".keras"),
        )
        ypred_last = deep_rgs_last.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred_last, datatype)

        deep_rgs_init = estimator_class()
        deep_rgs_init.load_model(
            model_path=os.path.join(tmp, init_file_name + ".keras"),
        )
        ypred_init = deep_rgs_init.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred_init, datatype)

        ypred = deep_rgs_train.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred, datatype)
        assert_array_almost_equal(ypred, ypred_best)


def check_regressor_train_estimate(estimator, datatype):
    """Test regressors that can produce train set prediction estimates."""
    estimator = _clone_estimator(estimator)
    estimator_class = type(estimator)

    # if we have a train_estimate parameter set use it, else use default
    if "_fit_predict" not in estimator_class.__dict__:
        raise ValueError(
            f"Regressor {estimator_class} has capability:train_estimate=True "
            "and must override the _fit_predict method."
        )

    # check the predictions are valid
    train_preds = estimator.fit_predict(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    _assert_predict_labels(train_preds, datatype, split="train")


def check_regressor_random_state_deep_learning(estimator, datatype):
    """Test deep regressor seeding."""
    random_state = 42

    deep_rgs1 = _clone_estimator(estimator, random_state=random_state)
    deep_rgs1.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    layers1 = deep_rgs1.training_model_.layers[1:]

    deep_rgs2 = _clone_estimator(estimator, random_state=random_state)
    deep_rgs2.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    layers2 = deep_rgs2.training_model_.layers[1:]

    assert len(layers1) == len(layers2)

    for i in range(len(layers1)):
        weights1 = layers1[i].get_weights()
        weights2 = layers2[i].get_weights()

        assert len(weights1) == len(weights2)

        for j in range(len(weights1)):
            _weight1 = np.asarray(weights1[j])
            _weight2 = np.asarray(weights2[j])

            np.testing.assert_almost_equal(_weight1, _weight2, 4)


def check_regressor_output(estimator, datatype):
    """Test regressor outputs the correct data types and values."""
    estimator = _clone_estimator(estimator)

    # run fit and predict
    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    _assert_predict_labels(y_pred, datatype)
