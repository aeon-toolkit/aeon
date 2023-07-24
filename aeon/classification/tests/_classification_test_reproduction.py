# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Functions for generating stored unit test results for classifiers."""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

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


if __name__ == "__main__":
    # _print_array(
    #     "ColumnEnsembleClassifier - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         ChannelEnsembleClassifier(
    #             estimators=[
    #                 (
    #                     "cBOSS",
    #                     ContractableBOSS(
    #                         n_parameter_samples=4,
    #                         max_ensemble_size=2,
    #                         random_state=0,
    #                         save_train_predictions=True,
    #                         feature_selection="none",
    #                     ),
    #                     [5],
    #                 ),
    #                 (
    #                     "CIF",
    #                     CanonicalIntervalForestClassifier(
    #                         n_estimators=2,
    #                         n_intervals=4,
    #                         att_subsample_size=4,
    #                         random_state=0,
    #                     ),
    #                     [3, 4],
    #                 ),
    #             ]
    #         )
    #     ),
    # )
    # _print_array(
    #     "BOSSEnsemble - UnitTest",
    #     _reproduce_classification_unit_test(
    #         BOSSEnsemble(max_ensemble_size=5, feature_selection="none", random_state=0)
    #     ),
    # )
    # _print_array(
    #     "ContractableBOSS - UnitTest",
    #     _reproduce_classification_unit_test(
    #         ContractableBOSS(
    #             n_parameter_samples=10,
    #             max_ensemble_size=5,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "MUSE - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         MUSE(
    #             window_inc=4,
    #             use_first_order_differences=False,
    #             support_probabilities=True,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "TemporalDictionaryEnsemble - UnitTest",
    #     _reproduce_classification_unit_test(
    #         TemporalDictionaryEnsemble(
    #             n_parameter_samples=10,
    #             max_ensemble_size=5,
    #             randomly_selected_params=5,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "TemporalDictionaryEnsemble - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         TemporalDictionaryEnsemble(
    #             n_parameter_samples=10,
    #             max_ensemble_size=5,
    #             randomly_selected_params=5,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "WEASEL - UnitTest",
    #     _reproduce_classification_unit_test(
    #         WEASEL(
    #             window_inc=4,
    #             random_state=0,
    #             support_probabilities=True,
    #             bigrams=False,
    #             feature_selection="none",
    #         )
    #     ),
    # )
    # _print_array(
    #     "WEASEL v2 - UnitTest",
    #     _reproduce_classification_unit_test(
    #         WEASEL_V2(
    #             random_state=0,
    #             support_probabilities=True,
    #             feature_selection="none",
    #         )
    #     ),
    # )
    # _print_array(
    #     "ElasticEnsemble - UnitTest",
    #     _reproduce_classification_unit_test(
    #         ElasticEnsemble(
    #             proportion_of_param_options=0.1,
    #             proportion_train_for_test=0.1,
    #             majority_vote=True,
    #             distance_measures=["dtw", "ddtw", "wdtw"],
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array("ShapeDTW - UnitTest", _reproduce_classification_unit_test(ShapeDTW()))
    # _print_array(
    #     "Catch22Classifier - UnitTest",
    #     _reproduce_classification_unit_test(
    #         Catch22Classifier(
    #             estimator=RandomForestClassifier(n_estimators=10),
    #             outlier_norm=True,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "Catch22Classifier - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         Catch22Classifier(
    #             estimator=RandomForestClassifier(n_estimators=10),
    #             outlier_norm=True,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "FreshPRINCEClassifier - UnitTest",
    #     _reproduce_classification_unit_test(
    #         FreshPRINCEClassifier(
    #             default_fc_parameters="minimal",
    #             n_estimators=10,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "FreshPRINCEClassifier - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         FreshPRINCEClassifier(
    #             default_fc_parameters="minimal",
    #             n_estimators=10,
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "MatrixProfileClassifier - UnitTest",
    #     _reproduce_classification_unit_test(
    #         MatrixProfileClassifier(subsequence_length=4, random_state=0)
    #     ),
    # )
    # _print_array(
    #     "RandomIntervalClassifier - UnitTest",
    #     _reproduce_classification_unit_test(
    #         RandomIntervalClassifier(
    #             n_intervals=3,
    #             interval_transformers=SummaryTransformer(
    #                 summary_function=("mean", "std", "min", "max"),
    #                 quantiles=(0.25, 0.5, 0.75),
    #             ),
    #             estimator=RandomForestClassifier(n_estimators=10),
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "RandomIntervalClassifier - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         RandomIntervalClassifier(
    #             n_intervals=3,
    #             interval_transformers=SummaryTransformer(
    #                 summary_function=("mean", "std", "min", "max"),
    #                 quantiles=(0.25, 0.5, 0.75),
    #             ),
    #             estimator=RandomForestClassifier(n_estimators=10),
    #             random_state=0,
    #         )
    #     ),
    # )
    # _print_array(
    #     "SignatureClassifier - UnitTest",
    #     _reproduce_classification_unit_test(
    #         SignatureClassifier(
    #             estimator=RandomForestClassifier(n_estimators=10), random_state=0
    #         )
    #     ),
    # )
    # _print_array(
    #     "SignatureClassifier - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         SignatureClassifier(
    #             estimator=RandomForestClassifier(n_estimators=10), random_state=0
    #         )
    #     ),
    # )
    # _print_array(
    #     "SummaryClassifier - UnitTest",
    #     _reproduce_classification_unit_test(
    #         SummaryClassifier(
    #             estimator=RandomForestClassifier(n_estimators=10), random_state=0
    #         )
    #     ),
    # )
    # _print_array(
    #     "SummaryClassifier - BasicMotions",
    #     _reproduce_classification_basic_motions(
    #         SummaryClassifier(
    #             estimator=RandomForestClassifier(n_estimators=10), random_state=0
    #         )
    #     ),
    # )
    # _print_array(
    #     "HIVECOTEV1 - UnitTest",
    #     _reproduce_classification_unit_test(
    #         HIVECOTEV1(
    #             stc_params={
    #                 "estimator": RandomForestClassifier(n_estimators=3),
    #                 "n_shapelet_samples": 50,
    #                 "max_shapelets": 5,
    #                 "batch_size": 10,
    #             },
    #             tsf_params={"n_estimators": 3},
    #             rise_params={"n_estimators": 3},
    #             cboss_params={"n_parameter_samples": 5, "max_ensemble_size": 3},
    #             random_state=0,
    #         )
    #     ),
    # )
    _print_array(
        "HIVECOTEV2 - UnitTest",
        _reproduce_classification_unit_test(
            HIVECOTEV2(
                stc_params={
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                drcif_params={
                    "n_estimators": 3,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                arsenal_params={"num_kernels": 50, "n_estimators": 3},
                tde_params={
                    "n_parameter_samples": 5,
                    "max_ensemble_size": 3,
                    "randomly_selected_params": 3,
                },
                random_state=0,
            )
        ),
    )
    _print_array(
        "HIVECOTEV2 - BasicMotions",
        _reproduce_classification_basic_motions(
            HIVECOTEV2(
                stc_params={
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                drcif_params={
                    "n_estimators": 3,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                arsenal_params={"num_kernels": 50, "n_estimators": 3},
                tde_params={
                    "n_parameter_samples": 5,
                    "max_ensemble_size": 3,
                    "randomly_selected_params": 3,
                },
                random_state=0,
            )
        ),
    )
    _print_array(
        "CanonicalIntervalForestClassifier - UnitTest",
        _reproduce_classification_unit_test(
            CanonicalIntervalForestClassifier(
                n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0
            )
        ),
    )
    _print_array(
        "CanonicalIntervalForestClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            CanonicalIntervalForestClassifier(
                n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0
            )
        ),
    )
    _print_array(
        "DrCIFClassifier - UnitTest",
        _reproduce_classification_unit_test(
            DrCIFClassifier(
                n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0
            )
        ),
    )
    _print_array(
        "DrCIFClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            DrCIFClassifier(
                n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0
            )
        ),
    )
    _print_array(
        "RandomIntervalSpectralEnsembleClassifier - UnitTest",
        _reproduce_classification_unit_test(
            RandomIntervalSpectralEnsembleClassifier(n_estimators=10, random_state=0)
        ),
    )
    _print_array(
        "SupervisedTimeSeriesForest - UnitTest",
        _reproduce_classification_unit_test(
            SupervisedTimeSeriesForest(n_estimators=10, random_state=0)
        ),
    )
    _print_array(
        "TimeSeriesForestClassifier - UnitTest",
        _reproduce_classification_unit_test(
            TimeSeriesForestClassifier(n_estimators=10, random_state=0)
        ),
    )
    _print_array(
        "Arsenal - UnitTest",
        _reproduce_classification_unit_test(
            Arsenal(num_kernels=20, n_estimators=5, random_state=0)
        ),
    )
    _print_array(
        "Arsenal - BasicMotions",
        _reproduce_classification_basic_motions(
            Arsenal(num_kernels=20, n_estimators=5, random_state=0)
        ),
    )
    _print_array(
        "RocketClassifier - UnitTest",
        _reproduce_classification_unit_test(
            RocketClassifier(num_kernels=100, random_state=0)
        ),
    )
    _print_array(
        "RocketClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            RocketClassifier(num_kernels=100, random_state=0)
        ),
    )
    _print_array(
        "ShapeletTransformClassifier - UnitTest",
        _reproduce_classification_unit_test(
            ShapeletTransformClassifier(
                estimator=RandomForestClassifier(n_estimators=5),
                n_shapelet_samples=50,
                max_shapelets=10,
                batch_size=10,
                random_state=0,
            )
        ),
    )
    _print_array(
        "ShapeletTransformClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            ShapeletTransformClassifier(
                estimator=RandomForestClassifier(n_estimators=5),
                n_shapelet_samples=50,
                max_shapelets=10,
                batch_size=10,
                random_state=0,
            )
        ),
    )

    _print_array(
        "ProbabilityThresholdEarlyClassifier - UnitTest",
        _reproduce_early_classification_unit_test(
            ProbabilityThresholdEarlyClassifier(
                random_state=0,
                classification_points=[6, 10, 16, 24],
                estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
            )
        ),
    )
    _print_array(
        "TEASER - UnitTest",
        _reproduce_early_classification_unit_test(
            TEASER(
                random_state=0,
                classification_points=[6, 10, 16, 24],
                estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
            )
        ),
    )
    _print_array(
        "TEASER-IF - UnitTest",
        _reproduce_early_classification_unit_test(
            TEASER(
                random_state=0,
                classification_points=[6, 10, 16, 24],
                estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
                one_class_classifier=IsolationForest(n_estimators=5),
                one_class_param_grid={"bootstrap": [True, False]},
            )
        ),
    )
