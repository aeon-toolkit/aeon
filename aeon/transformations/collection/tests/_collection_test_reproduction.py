# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""Functions for generating stored unit test results for transformers."""

import numpy as np

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.transformations.collection.catch22 import Catch22
from aeon.transformations.collection.random_intervals import RandomIntervals
from aeon.transformations.collection.shapelet_transform import RandomShapeletTransform
from aeon.transformations.collection.supervised_intervals import SupervisedIntervals


def _reproduce_transform_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(X_train), 5, replace=False)

    return np.nan_to_num(
        estimator.fit_transform(X_train[indices], y_train[indices]), False, 0, 0, 0
    )


def _reproduce_transform_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(X_train), 5, replace=False)

    return np.nan_to_num(
        estimator.fit_transform(X_train[indices], y_train[indices]), False, 0, 0, 0
    )


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
    _print_array(
        "Catch22 - UnitTest",
        _reproduce_transform_unit_test(Catch22(outlier_norm=True)),
    )
    _print_array(
        "Catch22 - BasicMotions",
        _reproduce_transform_basic_motions(Catch22()),
    )
    _print_array(
        "RandomIntervals - BasicMotions",
        _reproduce_transform_basic_motions(
            RandomIntervals(random_state=0, n_intervals=3)
        ),
    )
    _print_array(
        "SupervisedIntervals - BasicMotions",
        _reproduce_transform_basic_motions(
            SupervisedIntervals(
                random_state=0, n_intervals=1, randomised_split_point=True
            )
        ),
    )
    _print_array(
        "RandomShapeletTransform - UnitTest",
        _reproduce_transform_unit_test(
            RandomShapeletTransform(
                max_shapelets=10, n_shapelet_samples=500, random_state=0
            )
        ),
    )
    _print_array(
        "RandomShapeletTransform - BasicMotions",
        _reproduce_transform_basic_motions(
            RandomShapeletTransform(
                max_shapelets=10, n_shapelet_samples=500, random_state=0
            )
        ),
    )
