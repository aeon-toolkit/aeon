"""Test MinDist functions of symbolic representations."""

import numpy as np
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist._dft_sfa import mindist_dft_sfa_distance
from aeon.distances.mindist._paa_sax import mindist_paa_sax_distance
from aeon.distances.mindist._sax import mindist_sax_distance
from aeon.distances.mindist._sfa import mindist_sfa_distance
from aeon.transformations.collection.dictionary_based import SAX, SFA, SFAFast, SFAWhole


def test_sax_mindist():
    """Test the SAX Min-Distance function."""
    n_segments = 16
    alphabet_size = 8

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    SAX_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    SAX_train = SAX_transform.fit_transform(X_train).squeeze()
    PAA_train = SAX_transform._get_paa(X_train).squeeze()
    SAX_test = SAX_transform.transform(X_test).squeeze()

    for i in range(min(X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        # SAX Min-Distance
        mindist_sax = mindist_sax_distance(
            SAX_train[i], SAX_test[i], SAX_transform.breakpoints, X_train.shape[-1]
        )

        # SAX-PAA Min-Distance
        mindist_paa_sax = mindist_paa_sax_distance(
            PAA_train[i], SAX_test[i], SAX_transform.breakpoints, X_train.shape[-1]
        )

        # Euclidean Distance
        ed = np.linalg.norm(X[0] - Y[0])

        assert mindist_sax <= ed
        assert mindist_paa_sax >= mindist_sax  # a tighter lower bound
        assert mindist_paa_sax <= ed


def test_sfa_mindist():
    """Test the SFA Min-Distance function."""
    n_segments = 16
    alphabet_size = 8

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    n = X_train.shape[-1]
    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    histogram_type = "equi-width"

    sfa_fast = SFAFast(
        word_length=n_segments,
        alphabet_size=alphabet_size,
        window_size=n,
        binning_method=histogram_type,
        norm=True,
        variance=False,  # True gives a tighter lower bound
        lower_bounding_distances=True,  # This must be set!
    )

    sfa_old = SFA(
        word_length=n_segments,
        alphabet_size=alphabet_size,
        window_size=X_train.shape[-1],
        binning_method=histogram_type,
        norm=True,
        lower_bounding_distances=True,  # This must be set!
    )

    sfa_whole = SFAWhole(
        word_length=n_segments,
        alphabet_size=alphabet_size,
        binning_method=histogram_type,
        variance=False,  # True gives a tighter lower bound
        norm=True,
    )

    transforms = [sfa_fast, sfa_old, sfa_whole]
    dists = np.zeros(
        (min(X_train.shape[0], X_test.shape[0]), len(transforms)), dtype=np.float32
    )

    for j, sfa in enumerate(transforms):
        sfa.fit(X_train)
        X_train_words, X_train_dfts = sfa.transform_words(X_train)
        X_test_words, _ = sfa.transform_words(X_test)

        for i in range(min(X_train.shape[0], X_test.shape[0])):
            X = X_train[i].reshape(1, -1)
            Y = X_test[i].reshape(1, -1)

            # SFA Min-Distance
            mindist_sfa = mindist_sfa_distance(
                X_train_words[i], X_test_words[i], sfa.breakpoints
            )

            dists[i, j] = mindist_sfa

            # DFT-SFA Min-Distance
            mindist_dft_sfa = mindist_dft_sfa_distance(
                X_train_dfts[i], X_test_words[i], sfa.breakpoints
            )

            # Euclidean Distance
            ed = np.linalg.norm(X[0] - Y[0])

            assert mindist_sfa <= ed
            assert mindist_dft_sfa >= mindist_sfa  # a tighter lower bound
            assert mindist_dft_sfa <= ed

    for i in range(min(X_train.shape[0], X_test.shape[0])):
        assert np.allclose(*dists[i])


def test_sfa_whole_mindist():
    """Test the SFA Min-Distance function."""
    n_segments = 16
    alphabet_size = 8

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    histogram_type = "equi-width"

    sfa = SFAWhole(
        word_length=n_segments,
        alphabet_size=alphabet_size,
        binning_method=histogram_type,
        norm=True,
    )

    X_train_words, X_train_dfts = sfa.fit_transform(X_train)
    X_test_words, _ = sfa.transform(X_test)

    for i in range(min(X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        # SFA Min-Distance
        mindist_sfa = mindist_sfa_distance(
            X_train_words[i], X_test_words[i], sfa.breakpoints
        )

        # DFT-SFA Min-Distance
        mindist_dft_sfa = mindist_dft_sfa_distance(
            X_train_dfts[i], X_test_words[i], sfa.breakpoints
        )

        # Euclidean Distance
        ed = np.linalg.norm(X[0] - Y[0])

        assert mindist_sfa <= ed
        assert mindist_dft_sfa >= mindist_sfa  # a tighter lower bound
        assert mindist_dft_sfa <= ed
