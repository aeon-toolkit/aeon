"""Tests for all regressors."""

import os
import tempfile
import time
from functools import partial
from sys import platform

import numpy as np
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.datasets import load_cardano_sentiment, load_covid_3month
from aeon.regression.deep_learning import BaseDeepRegressor
from aeon.testing.expected_results.expected_regressor_outputs import (
    cardano_sentiment_preds,
    covid_3month_preds,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal


def _yield_regression_checks(estimator_class, estimator_instances, datatypes):
    """Yield all regression checks for an aeon regressor."""
    # only class required
    yield partial(
        check_regressor_against_expected_results, estimator_class=estimator_class
    )
    yield partial(check_regressor_tags_consistent, estimator_class=estimator_class)
    yield partial(
        check_regressor_does_not_override_final_methods, estimator_class=estimator_class
    )

    # data type irrelevant
    if issubclass(estimator_class, BaseDeepRegressor):
        yield partial(
            check_regressor_saving_loading_deep_learning,
            estimator_class=estimator_class,
            datatype=datatypes[0][0],
        )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # data type irrelevant
        if isinstance(estimator, BaseDeepRegressor):
            yield partial(
                check_regressor_random_state_deep_learning,
                estimator=estimator,
                datatype=datatypes[i][0],
            )


def check_regressor_against_expected_results(estimator_class):
    """Test classifier against stored results."""
    # we only use the first estimator instance for testing
    classname = estimator_class.__name__

    # We cannot guarantee same results on ARM macOS
    if platform == "darwin":
        return None

    for data_name, data_dict, data_loader, data_seed in [
        ["Covid3Month", covid_3month_preds, load_covid_3month, 0],
        ["CardanoSentiment", cardano_sentiment_preds, load_cardano_sentiment, 0],
    ]:
        # retrieve expected predict output, and skip test if not available
        if classname in data_dict.keys():
            expected_preds = data_dict[classname]
        else:
            # skip test if no expected preds are registered
            continue

        # we only use the first estimator instance for testing
        estimator_instance = estimator_class._create_test_instance(
            parameter_set="results_comparison"
        )
        # set random seed if possible
        set_random_state(estimator_instance, 0)

        # load test data
        X_train, y_train = data_loader(split="train")
        X_test, y_test = data_loader(split="test")
        indices_train = np.random.RandomState(data_seed).choice(
            len(y_train), 10, replace=False
        )
        indices_test = np.random.RandomState(data_seed).choice(
            len(y_test), 10, replace=False
        )

        # train regressor and predict
        estimator_instance.fit(X_train[indices_train], y_train[indices_train])
        y_pred = estimator_instance.predict(X_test[indices_test])

        # assert predictions are the same
        _assert_array_almost_equal(
            y_pred,
            expected_preds,
            decimal=2,
            err_msg=f"Failed to reproduce results for {classname} on {data_name}",
        )


def check_regressor_tags_consistent(estimator_class):
    """Test the tag X_inner_type is consistent with capability:unequal_length."""
    valid_types = {"np-list", "df-list", "pd-multivariate"}
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
        y = np.random.random(10)
        inst = estimator_class._create_test_instance(parameter_set="default")
        inst.fit(X, y)
        inst.predict(X)


def check_regressor_does_not_override_final_methods(estimator_class):
    """Test does not override final methods."""
    if "fit" in estimator_class.__dict__:
        raise ValueError(f"Classifier {estimator_class} overrides the method fit")
    if "predict" in estimator_class.__dict__:
        raise ValueError(
            f"Classifier {estimator_class} overrides the method " f"predict"
        )


def check_regressor_saving_loading_deep_learning(estimator_class, datatype):
    """Test Deep Regressor saving."""
    with tempfile.TemporaryDirectory() as tmp:
        if not (
            estimator_class.__name__
            in [
                "BaseDeepRegressor",
                "InceptionTimeRegressor",
                "LITETimeRegressor",
                "TapNetRegressor",
            ]
        ):
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
            ypred_best = deep_rgs_best.predict(
                FULL_TEST_DATA_DICT[datatype]["train"][0]
            )
            assert len(ypred_best) == len(FULL_TEST_DATA_DICT[datatype]["train"][1])

            deep_rgs_last = estimator_class()
            deep_rgs_last.load_model(
                model_path=os.path.join(tmp, last_file_name + ".keras"),
            )
            ypred_last = deep_rgs_last.predict(
                FULL_TEST_DATA_DICT[datatype]["train"][0]
            )
            assert len(ypred_last) == len(FULL_TEST_DATA_DICT[datatype]["train"][1])

            deep_rgs_init = estimator_class()
            deep_rgs_init.load_model(
                model_path=os.path.join(tmp, init_file_name + ".keras"),
            )
            ypred_init = deep_rgs_init.predict(
                FULL_TEST_DATA_DICT[datatype]["train"][0]
            )
            assert len(ypred_init) == len(FULL_TEST_DATA_DICT[datatype]["train"][1])


def check_regressor_random_state_deep_learning(estimator, datatype):
    """Test Deep Regressor seeding."""
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
