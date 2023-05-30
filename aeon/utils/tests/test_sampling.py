# -*- coding: utf-8 -*-
"""Testing sampling utilities."""

import numpy as np
import pandas as pd
import pytest

from aeon.utils._testing.deep_equals import deep_equals
from aeon.utils.sampling import random_partition, stratified_resample

NK_FIXTURES = [(10, 3), (15, 5), (19, 6), (3, 1), (1, 2)]
SEED_FIXTURES = [42, 0, 100, -5]
INPUT_TYPES = ["numpy3D", "np-list", "pd.DataFrame"]


@pytest.mark.parametrize("n, k", NK_FIXTURES)
def test_partition(n, k):
    """Test that random_partition returns a disjoint partition."""
    part = random_partition(n, k)

    assert isinstance(part, list)
    assert all(isinstance(x, list) for x in part)
    assert all(isinstance(x, int) for y in part for x in y)

    low_size = n // k
    hi_size = low_size + 1
    assert all(len(x) == low_size or len(x) == hi_size for x in part)

    part_union = set()
    for x in part:
        part_union = part_union.union(x)
    assert set(range(n)) == part_union

    for i, x in enumerate(part):
        for j, y in enumerate(part):
            if i != j:
                assert len(set(x).intersection(y)) == 0


@pytest.mark.parametrize("seed", SEED_FIXTURES)
@pytest.mark.parametrize("n, k", NK_FIXTURES)
def test_seed(n, k, seed):
    """Test that seed is deterministic."""
    part = random_partition(n, k, seed)
    part2 = random_partition(n, k, seed)

    assert deep_equals(part, part2)


@pytest.mark.parametrize("input_type", INPUT_TYPES)
def test_stratified_resample(input_type):
    random_state = np.random.RandomState(0)
    if input_type == "numpy3D":
        X_train = random_state.random((10, 1, 100))
        X_test = random_state.random((10, 1, 100))
    elif input_type == "np-list":
        X_train = [random_state.random((1, 100)) for _ in range(10)]
        X_test = [random_state.random((1, 100)) for _ in range(10)]
    else:
        train_series = [pd.Series(random_state.random(100)) for _ in range(10)]
        test_series = [pd.Series(random_state.random(100)) for _ in range(10)]
        X_train = pd.DataFrame({"dim_0": train_series})
        X_test = pd.DataFrame({"dim_0": test_series})
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_test = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    new_X_train, new_y_train, new_X_test, new_y_test = stratified_resample(
        X_train, y_train, X_test, y_test, random_state
    )

    # Valid return type
    assert type(X_train) == type(new_X_train) and type(X_test) == type(new_X_test)

    classes_train, classes_count_train = np.unique(y_train, return_counts=True)
    classes_test, classes_count_test = np.unique(y_test, return_counts=True)
    classes_new_train, classes_count_new_train = np.unique(
        new_y_train, return_counts=True
    )
    classes_new_test, classes_count_new_test = np.unique(new_y_test, return_counts=True)

    # Assert same class distributions
    assert np.all(classes_train == classes_new_train)
    assert np.all(classes_count_train == classes_count_new_train)
    assert np.all(classes_test == classes_new_test)
    assert np.all(classes_count_test == classes_count_new_test)
