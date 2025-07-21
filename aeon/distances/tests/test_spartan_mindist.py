"""Test MinDist functions of symbolic representations."""

import numpy as np
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist._pca_spartan import mindist_pca_spartan_distance
from aeon.distances.mindist._spartan import mindist_spartan_distance
from aeon.transformations.collection.dictionary_based import SPARTAN


def test_spartan_mindist():
    """Test the SPARTAN Min-Distance function."""
    n_segments = 16
    alphabet_size = 256

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    SPARTAN_transform = SPARTAN(
        word_length=n_segments, alphabet_size=alphabet_size, build_histogram=False
    )
    SPARTAN_train = SPARTAN_transform.fit_transform(X_train)
    SPARTAN_test = SPARTAN_transform.transform(X_test)
    _, SPARTAN_test_pca = SPARTAN_transform.transform_words(X_test)

    # print("alphabet size: ", SPARTAN_transform.alphabet_size)

    for i in range(min(X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        # SPARTAN Min-Distance
        mindist_spartan = mindist_spartan_distance(
            SPARTAN_test[i], SPARTAN_train[i], SPARTAN_transform.breakpoints
        )

        # SPARTAN Min-Distance
        mindist_pca_spartan = mindist_pca_spartan_distance(
            SPARTAN_test_pca[i], SPARTAN_train[i], SPARTAN_transform.breakpoints
        )

        # Euclidean Distance
        ed = np.linalg.norm(X[0] - Y[0])

        assert mindist_spartan <= mindist_pca_spartan
        assert mindist_pca_spartan <= ed
        assert mindist_spartan <= ed
