"""Functions to sample aeon datasets.

Used in experiments to get deterministic resamples.
"""

import random
from itertools import chain

import numpy as np
import pandas as pd
from deprecated.sphinx import deprecated
from sklearn.utils import check_random_state


# TODO: remove in v0.9.0
@deprecated(
    version="0.8.0",
    reason=(
        "stratified_resample is moving to "
        "benchmarking/experiments.py, "
        "this version will be removed in v0.9.0."
    ),
    category=FutureWarning,
)
def stratified_resample(X_train, y_train, X_test, y_test, random_state=None):
    """Stratified resample data without replacement using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints), list of
    shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i), or pd.DataFrame
        train data attributes.
    y_train : np.array
        train data class labels.
    X_test : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints), list of
    shape[n_cases] of 2D arrays shape (n_channels,n_timepoints_i), or pd.DataFrame
        test data attributes.
    y_test : np.array
        test data class labels as np array.
    random_state : int
        seed to enable reproduceable resamples

    Returns
    -------
    new train and test attributes and class labels.
    """
    random_state = check_random_state(random_state)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    if isinstance(X_train, pd.DataFrame):
        all_data = pd.concat([X_train, X_test], ignore_index=True)
    elif isinstance(X_train, list):
        all_data = list(x for x in chain(X_train, X_test))
    else:  # 3D or 2D numpy
        all_data = np.concatenate([X_train, X_test], axis=0)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test = np.unique(y_test)

    # haven't built functionality to deal with classes that exist in
    # test but not in train
    assert set(unique_train) == set(unique_test)

    new_train_indices = []
    new_test_indices = []
    for label, count_train in zip(unique_train, counts_train):
        class_indexes = np.argwhere(all_labels == label).ravel()

        # randomizes the order and partition into train and test
        random_state.shuffle(class_indexes)
        new_train_indices.extend(class_indexes[:count_train])
        new_test_indices.extend(class_indexes[count_train:])

    if isinstance(X_train, pd.DataFrame):
        new_X_train = all_data.iloc[new_train_indices]
        new_X_test = all_data.iloc[new_test_indices]
        new_X_train = new_X_train.reset_index(drop=True)
        new_X_test = new_X_test.reset_index(drop=True)
    elif isinstance(X_train, list):
        new_X_train = list(all_data[i] for i in new_train_indices)
        new_X_test = list(all_data[i] for i in new_test_indices)
    else:  # 3D or 2D numpy
        new_X_train = all_data[new_train_indices]
        new_X_test = all_data[new_test_indices]

    new_y_train = all_labels[new_train_indices]
    new_y_test = all_labels[new_test_indices]

    return new_X_train, new_y_train, new_X_test, new_y_test


def random_partition(n, k=2, seed=42):
    """Construct a uniformly random partition, iloc reference.

    Parameters
    ----------
    n : int
        size of set to partition
    k : int, optional, default=2
        number of sets to partition into
    seed : int
        random seed, used in random.shuffle

    Returns
    -------
    parts : list of list of int
        elements of `parts` are lists of iloc int indices between 0 and n-1
        elements of `parts` are of length floor(n / k) or ceil(n / k)
        elements of `parts`, as sets, are disjoint partition of [0, ..., n-1]
        elements of elements of `parts` are in no particular order
        `parts` is sampled uniformly at random, subject to the above properties
    """
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)

    parts = []
    for i in range(k):
        d = round(len(idx) / (k - i))
        parts += [idx[:d]]
        idx = idx[d:]

    return parts
