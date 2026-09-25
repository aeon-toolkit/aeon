"""Test MinDist functions of symbolic representations."""

import numpy as np
from scipy.stats import zscore

from aeon.datasets import load_unit_test
from aeon.distances.mindist._dft_sfa import (
    mindist_dft_sfa_distance,
    mindist_dft_sfa_pairwise_distance,
)
from aeon.distances.mindist._paa_sax import (
    mindist_paa_sax_distance,
    mindist_paa_sax_pairwise_distance,
)
from aeon.distances.mindist._sax import (
    mindist_sax_distance,
    mindist_sax_pairwise_distance,
)
from aeon.distances.mindist._sfa import (
    mindist_sfa_distance,
    mindist_sfa_pairwise_distance,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.dictionary_based import SAX, SFA, SFAFast, SFAWhole
from aeon.transformations.collection.dictionary_based._sfa_fast import (
    alphabet_allocation_methods,
)


def test_pairwise_mindist_matches_single_distances():
    """Test pairwise Mindist functions with 3D collection input."""
    X = np.array(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[3.0, 2.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0, 2.0]],
        ]
    )
    y = np.array(
        [
            [[1.0, 3.0, 1.0, 3.0]],
            [[2.0, 0.0, 2.0, 0.0]],
        ]
    )
    sax_breakpoints = np.array([0.5, 1.5, 2.5])
    sfa_breakpoints = np.tile(sax_breakpoints, (X.shape[-1], 1))
    symbolic_X = X.astype(np.int64)
    symbolic_y = y.astype(np.int64)

    functions = [
        (
            mindist_sax_pairwise_distance,
            mindist_sax_distance,
            symbolic_X,
            symbolic_y,
            sax_breakpoints,
            (8,),
        ),
        (
            mindist_sfa_pairwise_distance,
            mindist_sfa_distance,
            symbolic_X,
            symbolic_y,
            sfa_breakpoints,
            (),
        ),
        (
            mindist_paa_sax_pairwise_distance,
            mindist_paa_sax_distance,
            X,
            y,
            sax_breakpoints,
            (8,),
        ),
        (
            mindist_dft_sfa_pairwise_distance,
            mindist_dft_sfa_distance,
            X,
            y,
            sfa_breakpoints,
            (),
        ),
    ]

    for pairwise_distance, distance, x, y, breakpoints, extra_args in functions:
        pairwise = pairwise_distance(x, None, breakpoints, *extra_args)
        assert pairwise.shape == (len(x), len(x))
        for i in range(len(x)):
            for j in range(len(x)):
                expected = distance(
                    x[i].ravel(), x[j].ravel(), breakpoints, *extra_args
                )
                assert pairwise[i, j] == expected

        pairwise = pairwise_distance(x, y, breakpoints, *extra_args)
        assert pairwise.shape == (len(x), len(y))
        for i in range(len(x)):
            for j in range(len(y)):
                expected = distance(
                    x[i].ravel(), y[j].ravel(), breakpoints, *extra_args
                )
                assert pairwise[i, j] == expected


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


def test_single_sample():
    """Test the SFA Min-Distance function."""
    x, _ = make_example_3d_numpy(n_cases=1, n_channels=1, n_timepoints=10)
    y = x + 10
    transform = SFAWhole(word_length=8, alphabet_size=8, norm=True)
    x_sfa, _ = transform.fit_transform(x)
    _, y_dft = transform.transform(y)
    for i in range(len(x_sfa)):
        dist = mindist_dft_sfa_distance(y_dft[i], x_sfa[i], transform.breakpoints)
        assert dist == 0


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
        feature_selection_strategy=None,  # 'variance' gives a tighter lower bound
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
        feature_selection_strategy=None,  # 'variance' gives a tighter lower bound
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


def test_dynamic_alphabet_allocation():
    """Test the SFA Min-Distance function."""
    n_segments = 16
    alphabet_size = 64

    X_train, _ = load_unit_test("TRAIN")
    X_test, _ = load_unit_test("TEST")

    X_train = zscore(X_train.squeeze(), axis=1)
    X_test = zscore(X_test.squeeze(), axis=1)
    histogram_type = "equi-width"

    for alphabet_allocation_method in alphabet_allocation_methods:
        sfa = SFAWhole(
            word_length=n_segments,
            alphabet_size=alphabet_size,
            binning_method=histogram_type,
            alphabet_allocation_method=alphabet_allocation_method,
            feature_selection_strategy="variance",  # gives a tighter lower bound
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

            assert np.mean(np.log2(sfa.alphabet_sizes)) == np.log2(alphabet_size)
            assert mindist_sfa <= ed
            assert mindist_dft_sfa >= mindist_sfa  # a tighter lower bound
            assert mindist_dft_sfa <= ed
