"""Tests for all classifiers."""

import inspect
import os
import sys
import tempfile
import time
from copy import deepcopy
from functools import partial
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.ensemble._base import _set_random_states

from aeon.base import CheckpointableMixin
from aeon.base._base import _clone_estimator
from aeon.classification.deep_learning import BaseDeepClassifier
from aeon.testing.expected_results._write_estimator_results import (
    X_bm_test,
    X_bm_train,
    X_ut_test,
    X_ut_train,
    y_bm_train,
    y_ut_train,
)
from aeon.testing.expected_results.expected_classifier_results import (
    multivariate_expected_results,
    univariate_expected_results,
)
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.testing.utils.estimator_checks import (
    _assert_predict_labels,
    _assert_predict_probabilities,
    _get_tag,
)
from aeon.utils.data_types import COLLECTIONS_DATA_TYPES
from aeon.utils.validation.collection import get_n_cases


def _yield_classification_checks(estimator_class, estimator_instances, datatypes):
    """Yield all classification checks for an aeon classifier."""
    # only class required
    yield partial(
        check_classifier_against_expected_results,
        estimator_class=estimator_class,
        data_name="UnitTest",
        X_train=X_ut_train,
        y_train=y_ut_train,
        X_test=X_ut_test,
        results_dict=univariate_expected_results,
    )
    yield partial(
        check_classifier_against_expected_results,
        estimator_class=estimator_class,
        data_name="BasicMotions",
        X_train=X_bm_train,
        y_train=y_bm_train,
        X_test=X_bm_test,
        results_dict=multivariate_expected_results,
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

    if _get_tag(estimator_class, "capability:checkpointing", raise_error=True):
        yield partial(
            check_checkpointing_classifier,
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
                datatype=datatypes[i][0],
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
    estimator_class,
    data_name,
    X_train,
    y_train,
    X_test,
    results_dict,
):
    """Test classifier against stored results."""
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

    # train classifier and predict probas
    estimator_instance.fit(deepcopy(X_train), deepcopy(y_train))
    y_proba = estimator_instance.predict_proba(deepcopy(X_test))

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
        "resume_fit",
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


def check_checkpointing_classifier(estimator_class, datatype):
    """Compare uninterrupted training with recovery from a safe boundary.

    The ``checkpointing`` test parameter set must run for multiple batches with
    automatic checkpointing enabled. It should use a deterministic work limit,
    not wall-clock timing. Each implementation must additionally test its own
    continuation counters and random state.
    """
    import pytest

    assert issubclass(estimator_class, CheckpointableMixin)
    estimator = estimator_class._create_test_instance(parameter_set="checkpointing")
    _set_random_states(estimator, 42)
    X, y = FULL_TEST_DATA_DICT[datatype]["train"]
    X_test, _ = FULL_TEST_DATA_DICT[datatype]["test"]
    uninterrupted = estimator.clone().fit(X, y)

    class InterruptedFit(Exception):
        """Simulate job termination immediately after a safe checkpoint."""

    def interrupt_at_checkpoint(self, force=False):
        if not force:
            self.save_checkpoint(self.checkpoint_path)
            raise InterruptedFit

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "checkpoint.pkl")
        estimator.set_params(checkpoint_path=path, checkpoint_interval=1)
        with patch.object(
            estimator_class, "_checkpoint_if_due", interrupt_at_checkpoint
        ):
            with pytest.raises(InterruptedFit):
                estimator.fit(X, y)
        restored = estimator_class.load_checkpoint(path)
        assert not restored.is_fitted
        bad_y = y.copy()
        bad_y[0] = y[np.flatnonzero(y != y[0])[0]]
        with pytest.raises(ValueError, match="Unable to resume fitting: X and y"):
            restored.resume_fit(X, bad_y)
        restored.resume_fit(X, y)
        assert restored.is_fitted
        assert_array_almost_equal(
            uninterrupted.predict_proba(X_test), restored.predict_proba(X_test)
        )
        np.testing.assert_array_equal(
            uninterrupted.predict(X_test), restored.predict(X_test)
        )
        restored.fit(X, y)
        assert_array_almost_equal(
            uninterrupted.predict_proba(X_test), restored.predict_proba(X_test)
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
