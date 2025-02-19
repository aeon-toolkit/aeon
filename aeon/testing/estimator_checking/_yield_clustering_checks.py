"""Tests for all clusterers."""

import os
import tempfile
import time
from functools import partial

import numpy as np

from aeon.base._base import _clone_estimator
from aeon.clustering.deep_learning import BaseDeepClusterer
from aeon.testing.testing_data import FULL_TEST_DATA_DICT
from aeon.utils.validation import get_n_cases


def _yield_clustering_checks(estimator_class, estimator_instances, datatypes):
    """Yield all clustering checks for an aeon clusterer."""
    # only class required
    yield partial(check_clusterer_tags_consistent, estimator_class=estimator_class)
    yield partial(
        check_clusterer_does_not_override_final_methods, estimator_class=estimator_class
    )

    # test class instances
    for i, estimator in enumerate(estimator_instances):
        # data type irrelevant
        if isinstance(estimator, BaseDeepClusterer):
            yield partial(
                check_clustering_random_state_deep_learning,
                estimator=estimator,
                datatype=datatypes[i][0],
            )
        for datatype in datatypes[i]:
            yield partial(
                check_clusterer_output, estimator=estimator, datatype=datatype
            )

        if issubclass(estimator_class, BaseDeepClusterer):
            yield partial(
                check_clusterer_saving_loading_deep_learning,
                estimator_class=estimator_class,
                datatype=datatypes[i][0],
            )


def check_clusterer_tags_consistent(estimator_class):
    """Test all estimators capability tags reflect their capabilities."""
    # Test the tag X_inner_type is consistent with capability:unequal_length
    unequal_length = estimator_class.get_class_tag("capability:unequal_length")
    valid_types = {"np-list", "df-list", "pd-multivariate"}
    if unequal_length:  # one of X_inner_types must be capable of storing unequal length
        internal_types = estimator_class.get_class_tag("X_inner_type")
        if isinstance(internal_types, str):
            assert internal_types in valid_types
        else:  # must be a list
            assert bool(set(internal_types) & valid_types)
    # Test can actually fit/predict with multivariate if tag is set
    multivariate = estimator_class.get_class_tag("capability:multivariate")
    if multivariate:
        X = np.random.random((10, 2, 10))
        inst = estimator_class._create_test_instance(parameter_set="default")
        inst.fit(X)
        inst.predict(X)
        inst.predict_proba(X)


def check_clusterer_does_not_override_final_methods(estimator_class):
    """Test does not override final methods."""
    assert "fit" not in estimator_class.__dict__
    assert "predict" not in estimator_class.__dict__


def check_clustering_random_state_deep_learning(estimator, datatype):
    """Test Deep Clusterer seeding."""
    random_state = 42

    deep_clr1 = _clone_estimator(estimator, random_state=random_state)
    deep_clr1.fit(FULL_TEST_DATA_DICT[datatype]["train"][0])

    encoder_layers1 = deep_clr1.training_model_.layers[1].layers[1:]
    decoder_layers1 = deep_clr1.training_model_.layers[2].layers[1:]

    deep_clr2 = _clone_estimator(estimator, random_state=random_state)
    deep_clr2.fit(FULL_TEST_DATA_DICT[datatype]["train"][0])

    encoder_layers2 = deep_clr2.training_model_.layers[1].layers[1:]
    decoder_layers2 = deep_clr2.training_model_.layers[2].layers[1:]

    assert len(encoder_layers1) == len(encoder_layers2)
    assert len(decoder_layers1) == len(decoder_layers2)

    for i in range(len(encoder_layers1)):
        weights1 = encoder_layers1[i].get_weights()
        weights2 = encoder_layers2[i].get_weights()

        assert len(weights1) == len(weights2)

        for j in range(len(weights1)):
            _weight1 = np.asarray(weights1[j])
            _weight2 = np.asarray(weights2[j])

            np.testing.assert_almost_equal(_weight1, _weight2, 4)

    for i in range(len(decoder_layers1)):
        weights1 = decoder_layers1[i].get_weights()
        weights2 = decoder_layers2[i].get_weights()

        assert len(weights1) == len(weights2)

        for j in range(len(weights1)):
            _weight1 = np.asarray(weights1[j])
            _weight2 = np.asarray(weights2[j])

            np.testing.assert_almost_equal(_weight1, _weight2, 4)


def check_clusterer_output(estimator, datatype):
    """Test clusterer outputs the correct data types and values.

    Test predict produces a np.array or pd.Series with only values seen in the train
    data, and that predict_proba probability estimates add up to one.
    """
    estimator = _clone_estimator(estimator)

    # run fit and predict
    data = FULL_TEST_DATA_DICT[datatype]["train"][0]
    estimator.fit(data)
    assert hasattr(estimator, "labels_")
    assert isinstance(estimator.labels_, np.ndarray)
    assert np.array_equal(estimator.labels_, estimator.predict(data))

    y_pred = estimator.predict(FULL_TEST_DATA_DICT[datatype]["test"][0])

    # check predict
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (get_n_cases(FULL_TEST_DATA_DICT[datatype]["test"][0]),)

    # check predict proba (all classifiers have predict_proba by default)
    y_proba = estimator.predict_proba(FULL_TEST_DATA_DICT[datatype]["test"][0])

    assert isinstance(y_proba, np.ndarray)
    np.testing.assert_almost_equal(y_proba.sum(axis=1), 1, decimal=4)


def check_clusterer_saving_loading_deep_learning(estimator_class, datatype):
    """Test Deep Clusterer saving."""
    with tempfile.TemporaryDirectory() as tmp:
        if not (
            estimator_class.__name__
            in [
                "BaseDeepClusterer",
            ]
        ):
            if tmp[-1] != "/":
                tmp = tmp + "/"
            curr_time = str(time.time_ns())
            last_file_name = curr_time + "last"
            best_file_name = curr_time + "best"
            init_file_name = curr_time + "init"

            deep_cltr_train = estimator_class(
                **estimator_class._get_test_params()[0],
                save_best_model=True,
                save_last_model=True,
                save_init_model=True,
                best_file_name=best_file_name,
                last_file_name=last_file_name,
                init_file_name=init_file_name,
                file_path=tmp,
            )
            deep_cltr_train.fit(
                FULL_TEST_DATA_DICT[datatype]["train"][0],
                FULL_TEST_DATA_DICT[datatype]["train"][1],
            )

            estimator_pre_trained = deep_cltr_train._estimator

            deep_cltr_best = estimator_class()
            deep_cltr_best.load_model(
                model_path=os.path.join(tmp, best_file_name + ".keras"),
                estimator=estimator_pre_trained,
            )
            ypred_best = deep_cltr_best.predict(
                FULL_TEST_DATA_DICT[datatype]["train"][0]
            )
            assert len(ypred_best) == len(FULL_TEST_DATA_DICT[datatype]["train"][1])

            deep_cltr_last = estimator_class()
            deep_cltr_last.load_model(
                model_path=os.path.join(tmp, last_file_name + ".keras"),
                estimator=estimator_pre_trained,
            )
            ypred_last = deep_cltr_last.predict(
                FULL_TEST_DATA_DICT[datatype]["train"][0]
            )
            assert len(ypred_last) == len(FULL_TEST_DATA_DICT[datatype]["train"][1])

            deep_cltr_init = estimator_class()
            deep_cltr_init.load_model(
                model_path=os.path.join(tmp, init_file_name + ".keras"),
                estimator=estimator_pre_trained,
            )
            ypred_init = deep_cltr_init.predict(
                FULL_TEST_DATA_DICT[datatype]["train"][0]
            )
            assert len(ypred_init) == len(FULL_TEST_DATA_DICT[datatype]["train"][1])
