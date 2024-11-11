"""Functions for generating stored unit test results for classifiers."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.utils._testing import set_random_state

from aeon.classification import BaseClassifier
from aeon.classification.compose import ClassifierChannelEnsemble, ClassifierPipeline
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
    MrSQMClassifier,
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
    RDSTClassifier,
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
    if classifier_name == "ClassifierChannelEnsemble":
        classifier = ClassifierChannelEnsemble._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ClassifierPipeline":
        classifier = ClassifierPipeline._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "BOSSEnsemble":
        classifier = BOSSEnsemble._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ContractableBOSS":
        classifier = ContractableBOSS._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MUSE":
        classifier = MUSE._create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "TemporalDictionaryEnsemble":
        classifier = TemporalDictionaryEnsemble._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "WEASEL":
        classifier = WEASEL._create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "WEASEL_V2":
        classifier = WEASEL_V2._create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "REDCOMETS":
        classifier = REDCOMETS._create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "ElasticEnsemble":
        classifier = ElasticEnsemble._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "KNeighborsTimeSeriesClassifier":
        classifier = KNeighborsTimeSeriesClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "Catch22Classifier":
        classifier = Catch22Classifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "FreshPRINCEClassifier":
        classifier = FreshPRINCEClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RandomIntervalClassifier":
        classifier = RandomIntervalClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "QUANTClassifier":
        classifier = QUANTClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SignatureClassifier":
        classifier = SignatureClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SummaryClassifier":
        classifier = SummaryClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TSFreshClassifier":
        classifier = TSFreshClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HIVECOTEV1":
        classifier = HIVECOTEV1._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HIVECOTEV2":
        classifier = HIVECOTEV2._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "CanonicalIntervalForestClassifier":
        classifier = CanonicalIntervalForestClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "DrCIFClassifier":
        classifier = DrCIFClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "IntervalForestClassifier":
        classifier = IntervalForestClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RandomIntervalSpectralEnsembleClassifier":
        classifier = RandomIntervalSpectralEnsembleClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RSTSF":
        classifier = RSTSF._create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "SupervisedTimeSeriesForest":
        classifier = SupervisedTimeSeriesForest._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TimeSeriesForestClassifier":
        classifier = TimeSeriesForestClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "Arsenal":
        classifier = Arsenal._create_test_instance(parameter_set="results_comparison")
    elif classifier_name == "RocketClassifier":
        classifier = RocketClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "HydraClassifier":
        classifier = HydraClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MultiRocketHydraClassifier":
        classifier = MultiRocketHydraClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "OrdinalTDE":
        classifier = OrdinalTDE._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ShapeletTransformClassifier":
        classifier = ShapeletTransformClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "LearningShapeletClassifier":
        classifier = LearningShapeletClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RDSTClassifier":
        classifier = RDSTClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "MrSQMClassifier":
        classifier = MrSQMClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "SASTClassifier":
        classifier = SASTClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ContinuousIntervalTree":
        classifier = ContinuousIntervalTree._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "RotationForestClassifier":
        classifier = RotationForestClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "ProbabilityThresholdEarlyClassifier":
        classifier = ProbabilityThresholdEarlyClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "BaseEarlyClassifier":
        classifier = BaseEarlyClassifier._create_test_instance(
            parameter_set="results_comparison"
        )
    elif classifier_name == "TEASER":
        classifier = TEASER._create_test_instance(parameter_set="results_comparison")
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
