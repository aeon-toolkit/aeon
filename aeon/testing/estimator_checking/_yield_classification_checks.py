"""Tests for all classifiers."""

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
from aeon.classification.deep_learning import BaseDeepClassifier
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing.expected_results.expected_classifier_outputs import (
    basic_motions_proba,
    unit_test_proba,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import (
    _assert_predict_labels,
    _assert_predict_probabilities,
    _get_tag,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation import get_n_cases


def _yield_classification_checks(estimator_class, estimator_instances, datatypes):
    """Yield all classification checks for an aeon classifier."""
    # only class required
    if sys.platform == "linux":  # We cannot guarantee same results on ARM macOS
        # Compare against results for both UnitTest and BasicMotions if available
        yield partial(
            check_classifier_against_expected_results,
            estimator_class=estimator_class,
            data_name="UnitTest",
            data_loader=load_unit_test,
            results_dict=unit_test_proba,
            resample_seed=0,
        )
        yield partial(
            check_classifier_against_expected_results,
            estimator_class=estimator_class,
            data_name="BasicMotions",
            data_loader=load_basic_motions,
            results_dict=basic_motions_proba,
            resample_seed=4,
        )
    yield partial(check_classifier_overrides_and_tags, estimator_class=estimator_class)

    # data type irrelevant
    if _get_tag(estimator_class, "capability:contractable", raise_error=True):
        yield partial(
            check_contracted_classifier,
            estimator_class=estimator_class,
            datatype=datatypes[0][0],
        )

    if issubclass(estimator_class, BaseDeepClassifier):
        yield partial(
            check_classifier_saving_loading_deep_learning,
            estimator_class=estimator_class,
            datatype=datatypes[0][0],
        )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # data type irrelevant
        if _get_tag(estimator, "capability:train_estimate", raise_error=True):
            yield partial(
                check_classifier_train_estimate,
                estimator=estimator,
                datatype=datatypes[0][0],
            )

        if isinstance(estimator, BaseDeepClassifier):
            yield partial(
                check_classifier_random_state_deep_learning,
                estimator=estimator,
                datatype=datatypes[i][0],
            )

        # test all data types
        for datatype in datatypes[i]:
            yield partial(
                check_classifier_output, estimator=estimator, datatype=datatype
            )


def check_classifier_against_expected_results(
    estimator_class, data_name, data_loader, results_dict, resample_seed
):
    """Test classifier against stored results."""
    # retrieve expected predict_proba output, and skip test if not available
    if estimator_class.__name__ in results_dict.keys():
        expected_probas = results_dict[estimator_class.__name__]
    else:
        # skip test if no expected probas are registered
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
    indices = np.random.RandomState(resample_seed).choice(
        len(y_train), 10, replace=False
    )

    # train classifier and predict probas
    estimator_instance.fit(X_train[indices], y_train[indices])
    y_proba = estimator_instance.predict_proba(X_test[indices])

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


def check_classifier_overrides_and_tags(estimator_class):
    """Test compliance with the classifier base class contract."""
    # Test they don't override final methods, because Python does not enforce this
    final_methods = [
        "fit",
        "predict",
        "predict_proba",
        "fit_predict",
        "fit_predict_proba",
    ]
    for method in final_methods:
        if method in estimator_class.__dict__:
            raise ValueError(
                f"Classifier {estimator_class} overrides the method {method}. "
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


def check_contracted_classifier(estimator_class, datatype):
    """Test classifiers that can be contracted."""
    estimator_instance = estimator_class._create_test_instance(
        parameter_set="contracting"
    )
    default_params = inspect.signature(estimator_class.__init__).parameters

    # check that the classifier has a time_limit_in_minutes parameter
    if default_params.get("time_limit_in_minutes", None) is None:
        raise ValueError(
            f"Classifier {estimator_class} which sets "
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
            "Test parameters for test_contracted_classifier must set "
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
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)
    assert np.all(
        np.isin(np.unique(y_pred), np.unique(FULL_TEST_DATA_DICT[datatype]["test"][1]))
    )


def check_classifier_saving_loading_deep_learning(estimator_class, datatype):
    """Test deep classifier saving."""
    with tempfile.TemporaryDirectory() as tmp:
        if tmp[-1] != "/":
            tmp = tmp + "/"

        curr_time = str(time.time_ns())
        last_file_name = curr_time + "last"
        best_file_name = curr_time + "best"
        init_file_name = curr_time + "init"

        deep_cls_train = estimator_class(
            n_epochs=2,
            save_best_model=True,
            save_last_model=True,
            save_init_model=True,
            best_file_name=best_file_name,
            last_file_name=last_file_name,
            init_file_name=init_file_name,
            file_path=tmp,
        )
        deep_cls_train.fit(
            FULL_TEST_DATA_DICT[datatype]["train"][0],
            FULL_TEST_DATA_DICT[datatype]["train"][1],
        )

        deep_cls_best = estimator_class()
        deep_cls_best.load_model(
            model_path=os.path.join(tmp, best_file_name + ".keras"),
            classes=np.unique(FULL_TEST_DATA_DICT[datatype]["train"][1]),
        )
        ypred_best = deep_cls_best.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred_best, datatype)

        deep_cls_last = estimator_class()
        deep_cls_last.load_model(
            model_path=os.path.join(tmp, last_file_name + ".keras"),
            classes=np.unique(FULL_TEST_DATA_DICT[datatype]["train"][1]),
        )
        ypred_last = deep_cls_last.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred_last, datatype)

        deep_cls_init = estimator_class()
        deep_cls_init.load_model(
            model_path=os.path.join(tmp, init_file_name + ".keras"),
            classes=np.unique(FULL_TEST_DATA_DICT[datatype]["train"][1]),
        )
        ypred_init = deep_cls_init.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred_init, datatype)

        ypred = deep_cls_train.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
        _assert_predict_labels(ypred, datatype)
        assert_array_almost_equal(ypred, ypred_best)


def check_classifier_train_estimate(estimator, datatype):
    """Test classifiers that can produce train set probability estimates."""
    estimator = _clone_estimator(estimator)
    estimator_class = type(estimator)

    if (
        "_fit_predict" not in estimator_class.__dict__
        or "_fit_predict_proba" not in estimator_class.__dict__
    ):
        raise ValueError(
            f"Classifier {estimator_class} has capability:train_estimate=True "
            "and must override the _fit_predict and _fit_predict_proba methods."
        )

    unique_labels = np.unique(FULL_TEST_DATA_DICT[datatype]["train"][1])

    # check the predictions are valid
    train_preds = estimator.fit_predict(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    _assert_predict_labels(
        train_preds, datatype, split="train", unique_labels=unique_labels
    )

    # check the probabilities are valid
    train_proba = estimator.fit_predict_proba(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    _assert_predict_probabilities(
        train_proba, datatype, split="train", n_classes=len(unique_labels)
    )


def check_classifier_random_state_deep_learning(estimator, datatype):
    """Test deep classifier seeding."""
    random_state = 42

    deep_cls1 = _clone_estimator(estimator, random_state=random_state)
    deep_cls1.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    layers1 = deep_cls1.training_model_.layers[1:]

    deep_cls2 = _clone_estimator(estimator, random_state=random_state)
    deep_cls2.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )

    layers2 = deep_cls2.training_model_.layers[1:]

    assert len(layers1) == len(layers2)

    for i in range(len(layers1)):
        weights1 = layers1[i].get_weights()
        weights2 = layers2[i].get_weights()

        assert len(weights1) == len(weights2)

        for j in range(len(weights1)):
            _weight1 = np.asarray(weights1[j])
            _weight2 = np.asarray(weights2[j])

            np.testing.assert_almost_equal(_weight1, _weight2, 4)


def check_classifier_output(estimator, datatype):
    """Test classifier outputs the correct data types and values."""
    estimator = _clone_estimator(estimator)

    unique_labels = np.unique(FULL_TEST_DATA_DICT[datatype]["train"][1])

    # run fit and predict
    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])
    _assert_predict_labels(y_pred, datatype, unique_labels=unique_labels)

    # check predict proba (all classifiers have predict_proba by default)
    y_proba = estimator.predict_proba(FULL_TEST_DATA_DICT[datatype]["test"][0])
    _assert_predict_probabilities(y_proba, datatype, n_classes=len(unique_labels))
