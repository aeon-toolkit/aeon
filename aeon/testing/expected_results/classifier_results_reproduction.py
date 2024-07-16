"""Functions for generating stored unit test results for classifiers."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.utils._testing import set_random_state

from aeon.classification import BaseClassifier
from aeon.classification.compose import (
    ChannelEnsembleClassifier,
    ClassifierPipeline,
    WeightedEnsembleClassifier,
)
from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)
from aeon.classification.dictionary_based import (
    MUSE,
    REDCOMETS,
    WEASEL,
    WEASEL_V2,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
)
from aeon.classification.distance_based import (
    ElasticEnsemble,
    KNeighborsTimeSeriesClassifier,
)
from aeon.classification.early_classification import (
    TEASER,
    BaseEarlyClassifier,
    ProbabilityThresholdEarlyClassifier,
)
from aeon.classification.feature_based import (
    Catch22Classifier,
    FreshPRINCEClassifier,
    SignatureClassifier,
    SummaryClassifier,
    TSFreshClassifier,
)
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from aeon.classification.interval_based import (
    RSTSF,
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    IntervalForestClassifier,
    QUANTClassifier,
    RandomIntervalClassifier,
    RandomIntervalSpectralEnsembleClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from aeon.classification.ordinal_classification import OrdinalTDE
from aeon.classification.shapelet_based import (
    LearningShapeletClassifier,
    MrSQMClassifier,
    SASTClassifier,
    ShapeletTransformClassifier,
)
from aeon.classification.sklearn import ContinuousIntervalTree, RotationForestClassifier
from aeon.datasets import load_basic_motions, load_unit_test


def _reproduce_classification_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    X_test, _ = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train[indices], y_train[indices])
    return estimator.predict_proba(X_test[indices])


def _reproduce_classification_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    X_test, _ = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train[indices], y_train[indices])
    return estimator.predict_proba(X_test[indices])


def _reproduce_early_classification_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    X_test, _ = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train[indices], y_train[indices])
    return estimator.predict_proba(X_test[indices])[0]


def _reproduce_early_classification_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    X_test, _ = load_basic_motions(split="test")
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


def _print_results_for_classifier(classifier_name, dataset_name):
    if classifier_name == "ChannelEnsembleClassifier":
        classifier = ChannelEnsembleClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "WeightedEnsembleClassifier":
        classifier = WeightedEnsembleClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ClassifierPipeline":
        classifier = ClassifierPipeline.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "BOSSEnsemble":
        classifier = BOSSEnsemble.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ContractableBOSS":
        classifier = ContractableBOSS.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MUSE":
        classifier = MUSE.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "TemporalDictionaryEnsemble":
        classifier = TemporalDictionaryEnsemble.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "WEASEL":
        classifier = WEASEL.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "WEASEL_V2":
        classifier = WEASEL_V2.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "REDCOMETS":
        classifier = REDCOMETS.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "ElasticEnsemble":
        classifier = ElasticEnsemble.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "KNeighborsTimeSeriesClassifier":
        classifier = KNeighborsTimeSeriesClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "Catch22Classifier":
        classifier = Catch22Classifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "FreshPRINCEClassifier":
        classifier = FreshPRINCEClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RandomIntervalClassifier":
        classifier = RandomIntervalClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "QUANTClassifier":
        classifier = QUANTClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SignatureClassifier":
        classifier = SignatureClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SummaryClassifier":
        classifier = SummaryClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TSFreshClassifier":
        classifier = TSFreshClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HIVECOTEV1":
        classifier = HIVECOTEV1.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "HIVECOTEV2":
        classifier = HIVECOTEV2.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "CanonicalIntervalForestClassifier":
        classifier = CanonicalIntervalForestClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "DrCIFClassifier":
        classifier = DrCIFClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "IntervalForestClassifier":
        classifier = IntervalForestClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RandomIntervalSpectralEnsembleClassifier":
        classifier = RandomIntervalSpectralEnsembleClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RSTSF":
        classifier = RSTSF.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "SupervisedTimeSeriesForest":
        classifier = SupervisedTimeSeriesForest.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TimeSeriesForestClassifier":
        classifier = TimeSeriesForestClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "Arsenal":
        classifier = Arsenal.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "RocketClassifier":
        classifier = RocketClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HydraClassifier":
        classifier = HydraClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MultiRocketHydraClassifier":
        classifier = MultiRocketHydraClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "OrdinalTDE":
        classifier = OrdinalTDE.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "ShapeletTransformClassifier":
        classifier = ShapeletTransformClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "LearningShapeletClassifier":
        classifier = LearningShapeletClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MrSQMClassifier":
        classifier = MrSQMClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SASTClassifier":
        classifier = SASTClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ContinuousIntervalTree":
        classifier = ContinuousIntervalTree.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RotationForestClassifier":
        classifier = RotationForestClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ProbabilityThresholdEarlyClassifier":
        classifier = ProbabilityThresholdEarlyClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "BaseEarlyClassifier":
        classifier = BaseEarlyClassifier.create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TEASER":
        classifier = TEASER.create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "TEASER-IF":
        classifier = TEASER(
            classification_points=[6, 10, 16, 24],
            estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
            one_class_classifier=IsolationForest(n_estimators=5, random_state=0),
            one_class_param_grid={"bootstrap": [True, False]},
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    if dataset_name == "UnitTest":
        data_function = (
            _reproduce_classification_unit_test
            if isinstance(classifier, BaseClassifier)
            else _reproduce_early_classification_unit_test
        )
    elif dataset_name == "BasicMotions":
        data_function = (
            _reproduce_classification_basic_motions
            if isinstance(classifier, BaseClassifier)
            else _reproduce_early_classification_basic_motions
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    set_random_state(classifier, 0)

    _print_array(
        f"{classifier_name} - {dataset_name}",
        data_function(classifier),
    )


if __name__ == "__main__":
    # change as required when adding new classifiers, datasets or updating results
    _print_results_for_classifier("HIVECOTEV2", "BasicMotions")
