"""Test MinDist functions of symbolic representations."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances._dft_sfa_mindist import dft_sfa_mindist
from aeon.distances._paa_sax_mindist import paa_sax_mindist
from aeon.distances._sax_mindist import sax_mindist
from aeon.distances._sfa_mindist import sfa_mindist
from aeon.transformations.collection.dictionary_based import SAX, SFA, SFAFast


def test_sax_mindist():
    n_segments = 16
    alphabet_size = 8

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    SAX_transform = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    SAX_train = SAX_transform.fit_transform(X_train).squeeze()
    SAX_test = SAX_transform.transform(X_test).squeeze()
    PAA_train = SAX_transform._get_paa(X_train).squeeze()

    tightness_sax = 0.0
    tightness_paa_sax = 0.0
    for i in range(min(X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        # SAX Min-Distance
        mindist_sax = sax_mindist(
            SAX_train[i], SAX_test[i], SAX_transform.breakpoints, X_train.shape[-1]
        )

        # SAX-PAA Min-Distance
        mindist_paa_sax = paa_sax_mindist(
            PAA_train[i], SAX_test[i], SAX_transform.breakpoints, X_train.shape[-1]
        )

        # Euclidean Distance
        ed = np.linalg.norm(X[0] - Y[0])

        if ed > 0:
            tightness_sax += mindist_sax / ed
            tightness_paa_sax += mindist_paa_sax / ed

        if mindist_sax > ed:
            print("mindist_sax is:\t", mindist_sax)
            print("ED is:         \t", ed)

        if mindist_paa_sax > ed:
            print("mindist_paa_sax is:\t", mindist_paa_sax, mindist_sax)
            print("ED is:             \t", ed)

    print("\n")
    print("All ok:")
    print("PAA-SAX-Min-Dist Tightness:\t", tightness_paa_sax / X_train.shape[0])
    print("SAX-Min-Dist Tightness:    \t", tightness_sax / X_train.shape[0])


def test_sfa_mindist():
    n_segments = 32
    alphabet_size = 8

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    n = X_train.shape[-1]
    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)

    histogram = "equi-width"

    sfa = SFAFast(
        word_length=n_segments,
        alphabet_size=alphabet_size,
        window_size=n,
        binning_method=histogram,
        norm=True,
        variance=True,
        lower_bounding=True,
    )

    # sfa = SFA(
    #       word_length=n_segments,
    #       alphabet_size=alphabet_size,
    #       window_size=X_train.shape[-1],
    #       binning_method=histogram,
    #       norm=True,
    #       lower_bounding=True,
    # )

    sfa.fit(X_train)
    X_train_words = sfa.transform_words(X_train)
    Y_train_words = sfa.transform_words(X_test)

    SFA_train_dfts = sfa.transform_mft(X_train).squeeze()

    tightness_sfa = 0.0
    tightness_dft_sfa = 0.0
    for i in range(min(X_train.shape[0], X_test.shape[0])):
        X = X_train[i].reshape(1, -1)
        Y = X_test[i].reshape(1, -1)

        # SFA Min-Distance
        mindist_sfa = sfa_mindist(X_train_words[i], Y_train_words[i], sfa.breakpoints)

        # DFT-SFA Min-Distance
        mindist_dft_sfa = dft_sfa_mindist(
            SFA_train_dfts[i], Y_train_words[i], sfa.breakpoints
        )

        # Euclidean Distance
        ed = np.linalg.norm(X[0] - Y[0])

        if ed > 0:
            tightness_sfa += mindist_sfa / ed
            tightness_dft_sfa += mindist_dft_sfa / ed

        if mindist_sfa > ed:
            print("mindist_sfa is:\t", mindist_sfa)
            print("ED is:         \t", ed)

        if mindist_dft_sfa > ed:
            print("mindist_paa_sax is:\t", mindist_dft_sfa, mindist_sfa)
            print("ED is:             \t", ed)

    print("\n")
    print("All ok:")
    print("DFT-SFA-Min-Dist Tightness:\t", tightness_dft_sfa / X_train.shape[0])
    print("SFA-Min-Dist Tightness:    \t", tightness_sfa / X_train.shape[0])
