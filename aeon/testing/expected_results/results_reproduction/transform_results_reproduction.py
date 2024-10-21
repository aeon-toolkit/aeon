"""Functions for generating stored unit test results for transformers."""

import numpy as np
from sklearn.utils._testing import set_random_state

from aeon.datasets import load_basic_motions, load_unit_test
from aeon.transformations.collection.interval_based import (
    RandomIntervals,
    SupervisedIntervals,
)
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform


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


def _print_results_for_transformer(transformer_name, dataset_name):
    if transformer_name == "RandomIntervals":
        transformer = RandomIntervals._create_test_instance(
            parameter_set="results_comparison"
        )
    elif transformer_name == "SupervisedIntervals":
        transformer = SupervisedIntervals._create_test_instance(
            parameter_set="results_comparison"
        )
    elif transformer_name == "RandomShapeletTransform":
        transformer = RandomShapeletTransform._create_test_instance(
            parameter_set="results_comparison"
        )
    else:
        raise ValueError(f"Unknown transformer: {transformer_name}")

    if dataset_name == "UnitTest":
        data_function = _reproduce_transform_unit_test
    elif dataset_name == "BasicMotions":
        data_function = _reproduce_transform_basic_motions
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    set_random_state(transformer, 0)

    _print_array(
        f"{transformer_name} - {dataset_name}",
        data_function(transformer),
    )


if __name__ == "__main__":
    # change as required when adding new transformers, datasets or updating results
    _print_results_for_transformer("RandomIntervals", "BasicMotions")
