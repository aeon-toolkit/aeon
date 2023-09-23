# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Functions for generating stored unit test results for classifiers."""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.utils._testing import set_random_state

from aeon.classification import BaseClassifier
from aeon.classification.compose import ChannelEnsembleClassifier
from aeon.classification.convolution_based import Arsenal, RocketClassifier
from aeon.classification.dictionary_based import (
    MUSE,
    WEASEL,
    WEASEL_V2,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
)
from aeon.classification.distance_based import ElasticEnsemble, ShapeDTW
from aeon.classification.early_classification import (
    TEASER,
    ProbabilityThresholdEarlyClassifier,
)
from aeon.classification.feature_based import (
    Catch22Classifier,
    FreshPRINCEClassifier,
    MatrixProfileClassifier,
    SignatureClassifier,
    SummaryClassifier,
)
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from aeon.classification.interval_based import (
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    RandomIntervalClassifier,
    RandomIntervalSpectralEnsembleClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.transformations.series.summarize import SummaryTransformer


def _reproduce_classification_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train, y_train)
    return estimator.predict_proba(X_test[indices])


def _reproduce_classification_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train[indices], y_train[indices])
    return estimator.predict_proba(X_test[indices])


def _reproduce_early_classification_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train, y_train)
    return estimator.predict_proba(X_test[indices])[0]


def _reproduce_early_classification_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train[indices], y_train[indices])
    return estimator.predict_proba(X_test[indices])[0]


# flake8: noqa: T001
def _print_array(test_name, array):
    print(test_name)
    print("[")
    for sub_array in array:
        print("[", end="")
        for i, value in enumerate(sub_array):
            print(str(round(value, 4)), end="")
            if i < len(sub_array) - 1:
                print(", ", end="")
        print("],")
    print("]")


def _print_results_for_transformer(classifier_name, dataset_name):
    if classifier_name == "ChannelEnsembleClassifier":
        transformer = ChannelEnsembleClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "BOSSEnsemble":
        transformer = BOSSEnsemble.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ContractableBOSS":
        transformer = ContractableBOSS.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MUSE":
        transformer = MUSE.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "TemporalDictionaryEnsemble":
        transformer = TemporalDictionaryEnsemble.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "WEASEL":
        transformer = WEASEL.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "WEASEL_V2":
        transformer = WEASEL_V2.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "ElasticEnsemble":
        transformer = ElasticEnsemble.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ShapeDTW":
        transformer = ShapeDTW.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "Catch22Classifier":
        transformer = Catch22Classifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "FreshPRINCEClassifier":
        transformer = FreshPRINCEClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MatrixProfileClassifier":
        transformer = MatrixProfileClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RandomIntervalClassifier":
        transformer = RandomIntervalClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SignatureClassifier":
        transformer = SignatureClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SummaryClassifier":
        transformer = SummaryClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HIVECOTEV1":
        transformer = HIVECOTEV1.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HIVECOTEV2":
        transformer = HIVECOTEV2.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "CanonicalIntervalForestClassifier":
        transformer = CanonicalIntervalForestClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "DrCIFClassifier":
        transformer = DrCIFClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RandomIntervalSpectralEnsembleClassifier":
        transformer = RandomIntervalSpectralEnsembleClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SupervisedTimeSeriesForest":
        transformer = SupervisedTimeSeriesForest.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TimeSeriesForestClassifier":
        transformer = TimeSeriesForestClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "Arsenal":
        transformer = Arsenal.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "RocketClassifier":
        transformer = RocketClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ShapeletTransformClassifier":
        transformer = ShapeletTransformClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ProbabilityThresholdEarlyClassifier":
        transformer = ProbabilityThresholdEarlyClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TEASER":
        transformer = TEASER.create_test_instance(parameter_set="results_comparison")
    else:
        raise ValueError(f"Unknown transformer: {classifier_name}")

    if dataset_name == "UnitTest":
        data_function = (
            _reproduce_classification_unit_test
            if isinstance(transformer, BaseClassifier)
            else _reproduce_early_classification_unit_test
        )
    elif dataset_name == "BasicMotions":
        data_function = (
            _reproduce_classification_basic_motions
            if isinstance(transformer, BaseClassifier)
            else _reproduce_early_classification_basic_motions
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    set_random_state(transformer, 0)

    _print_array(
        f"{classifier_name} - {dataset_name}",
        data_function(transformer),
    )


if __name__ == "__main__":
    # change as required when adding new transformers, datasets or updating results
    _print_results_for_transformer("TEASER", "UnitTest")
