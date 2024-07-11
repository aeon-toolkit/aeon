import inspect
from functools import partial
from sys import platform

import numpy as np
from sklearn.utils._testing import set_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.deep_learning import BaseDeepClassifier
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.testing.expected_results.expected_classifier_outputs import (
    basic_motions_proba,
    unit_test_proba,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import _assert_array_almost_equal, _get_tag
from aeon.utils.validation import get_n_cases


def _yield_classification_checks(estimator_class, estimator_instances, datatypes):
    """Yield all classification checks for an aeon classifier."""
    # no data needed
    yield test_classifier_against_expected_results
    yield test_classifier_tags_consistent
    yield test_does_not_override_final_methods

    # data type irrelevant
    if _get_tag(estimator, "capability:contractable"):
        yield partial(test_contracted_classifier, datatype=datatypes[0])

    if _get_tag(estimator, "capability:train_estimate"):
        yield partial(test_classifier_train_estimate, datatype=datatypes[0])

    if isinstance(estimator, BaseDeepClassifier):
        yield partial(test_random_state_deep_learning_cls, datatype=datatypes[0])

    # test all data types
    for datatype in datatypes:
        yield partial(test_classifier_output, datatype=datatype)


def test_classifier_against_expected_results(estimator):
    """Test classifier against stored results."""
    # we only use the first estimator instance for testing

    #todo remove
    return None

    class_name = type(estimator).__name__

    # We cannot guarantee same results on ARM macOS
    if platform == "darwin":
        return None

    # the test currently fails when numba is disabled. See issue #622
    import os

    if class_name == "HIVECOTEV2" and os.environ.get("NUMBA_DISABLE_JIT") == "1":
        return None

    for data_name, data_dict, data_loader, data_seed in [
        ["UnitTest", unit_test_proba, load_unit_test, 0],
        ["BasicMotions", basic_motions_proba, load_basic_motions, 4],
    ]:
        # retrieve expected predict_proba output, and skip test if not available
        if class_name in data_dict.keys():
            expected_probas = data_dict[class_name]
        else:
            # skip test if no expected probas are registered
            continue

        # we only use the first estimator instance for testing
        estimator_instance = estimator.create_test_instance(
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
            err_msg=f"Failed to reproduce results for {class_name} on {data_name}",
        )


def test_classifier_tags_consistent(estimator):
    """Test the tag X_inner_type is consistent with capability:unequal_length."""
    estimator_class = type(estimator)
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


def test_does_not_override_final_methods(estimator):
    """Test does not override final methods."""
    estimator_class = type(estimator)
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


def test_contracted_classifier(estimator, datatype):
    """Test classifiers that can be contracted."""
    estimator_class = type(estimator)

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
    if vars(estimator).get("time_limit_in_minutes", None) < 0.5:
        raise ValueError(
            "Test parameters for test_contracted_classifier must set "
            "time_limit_in_minutes to 0.5 or more. It is recommended to make "
            "this larger and add an alternative stopping mechanism "
            "(i.e. max ensemble members)."
        )

    # run fit and predict
    estimator.fit(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])

    # check predict
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)
    assert np.all(
        np.isin(np.unique(y_pred), np.unique(FULL_TEST_DATA_DICT[datatype]["test"][1]))
    )


def test_classifier_train_estimate(estimator, datatype):
    """Test classifiers that can produce train set probability estimates."""
    estimator = _clone_estimator(estimator)
    estimator_class = type(estimator)

    # if we have a train_estimate parameter set use it, else use default
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
    assert isinstance(train_preds, np.ndarray)
    assert train_preds.shape == (
        get_n_cases(FULL_TEST_DATA_DICT[datatype]["train"][0]),
    )
    assert np.all(np.isin(np.unique(train_preds), unique_labels))

    # check the probabilities are valid
    train_proba = estimator.fit_predict_proba(
        FULL_TEST_DATA_DICT[datatype]["train"][0],
        FULL_TEST_DATA_DICT[datatype]["train"][1],
    )
    assert isinstance(train_proba, np.ndarray)
    assert train_proba.shape == (
        get_n_cases(FULL_TEST_DATA_DICT[datatype]["train"][0]),
        len(unique_labels),
    )
    np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)


def test_random_state_deep_learning_cls(estimator, datatype):
    """Test Deep Classifier seeding."""
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


def test_classifier_output(estimator, datatype):
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
    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])

    # check predict
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)
    assert np.all(np.isin(np.unique(y_pred), unique_labels))

    # check predict proba (all classifiers have predict_proba by default)
    y_proba = estimator.predict_proba(FULL_TEST_DATA_DICT[datatype]["test"][0])

    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape == (
        get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),
        len(unique_labels),
    )
    np.testing.assert_almost_equal(y_proba.sum(axis=1), 1, decimal=4)
