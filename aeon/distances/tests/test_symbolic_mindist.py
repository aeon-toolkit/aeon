"""Test MinDist functions of symbolic representations."""

import numpy as np
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist._dft_sfa import mindist_dft_sfa_distance
from aeon.distances.mindist._paa_sax import mindist_paa_sax_distance
from aeon.distances.mindist._sax import mindist_sax_distance
from aeon.distances.mindist._sfa import mindist_sfa_distance
from aeon.transformations.collection.dictionary_based import SAX, SFA, SFAFast


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
        variance=True,
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
    transforms = [sfa_old, sfa_fast]

    for sfa in transforms:
        sfa.fit(X_train)
        X_train_words = sfa.transform_words(X_train).squeeze()
        Y_train_words = sfa.transform_words(X_test).squeeze()

        SFA_train_dfts = sfa.transform_mft(X_train).squeeze()

        for i in range(min(X_train.shape[0], X_test.shape[0])):
            X = X_train[i].reshape(1, -1)
            Y = X_test[i].reshape(1, -1)

            # SFA Min-Distance
            mindist_sfa = mindist_sfa_distance(
                X_train_words[i], Y_train_words[i], sfa.breakpoints
            )

            # DFT-SFA Min-Distance
            mindist_dft_sfa = mindist_dft_sfa_distance(
                SFA_train_dfts[i], Y_train_words[i], sfa.breakpoints
            )

            # Euclidean Distance
            ed = np.linalg.norm(X[0] - Y[0])

            assert mindist_sfa <= ed
            assert mindist_dft_sfa >= mindist_sfa  # a tighter lower bound
            assert mindist_dft_sfa <= ed
