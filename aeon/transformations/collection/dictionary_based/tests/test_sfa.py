"""Test for SFA transformer."""

import sys

import numpy as np
import pytest

from aeon.datasets import load_unit_test
from aeon.testing.data_generation import (
    make_example_3d_numpy,
)
from aeon.transformations.collection.dictionary_based import SFA, SFAFast


@pytest.mark.parametrize(
    "binning_method", ["equi-depth", "equi-width", "information-gain", "kmeans"]
)
def test_transformer(binning_method):
    """Check the transformer has changed the data correctly."""
    # load training data
    X = np.random.rand(10, 1, 150)
    y = np.random.randint(0, 2, 10)

    word_length = 6
    alphabet_size = 4

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        binning_method=binning_method,
    )
    p.fit(X, y)

    assert p.breakpoints.shape == (word_length, alphabet_size)
    i = sys.float_info.max if binning_method == "information-gain" else 0
    assert np.equal(i, p.breakpoints[1, :-1]).all()  # imag component is 0 or inf
    _ = p.transform(X, y)


@pytest.mark.parametrize("use_fallback_dft", [True, False])
@pytest.mark.parametrize("norm", [True, False])
def test_dft_mft(use_fallback_dft, norm):
    """Test the DFT and MFT of the SFA transformer."""
    # load training data
    X = np.random.rand(10, 1, 150)
    y = np.random.randint(0, 2, 10)
    X_tab = X.squeeze()

    word_length = 6
    alphabet_size = 4

    # Single DFT transformation
    window_size = X_tab.shape[1]

    p = SFA(
        word_length=6,
        alphabet_size=4,
        window_size=window_size,
        norm=norm,
        use_fallback_dft=use_fallback_dft,
    ).fit(X, y)

    if use_fallback_dft:
        dft = p._discrete_fourier_transform(X_tab[0], word_length, norm, 1, True)
    else:
        dft = p._fast_fourier_transform(X_tab[0])

    mft = p._mft(X_tab[0])

    assert (mft - dft < 0.0001).all()

    # Windowed DFT transformation
    window_size = 140

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        window_size=window_size,
        norm=norm,
        use_fallback_dft=use_fallback_dft,
    ).fit(X, y)

    mft = p._mft(X_tab[0])
    for i in range(len(X_tab[0]) - window_size + 1):
        if use_fallback_dft:
            dft = p._discrete_fourier_transform(
                X_tab[0, i : window_size + i], word_length, norm, 1, True
            )
        else:
            dft = p._fast_fourier_transform(X_tab[0, i : window_size + i])

        assert (mft[i] - dft < 0.001).all()

    assert len(mft) == len(X_tab[0]) - window_size + 1
    assert len(mft[0]) == word_length


@pytest.mark.parametrize("binning_method", ["equi-depth", "information-gain"])
def test_sfa_anova(binning_method):
    """Test the SFA transformer with ANOVA one-sided test."""
    # load training data
    X = np.random.rand(10, 1, 150)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    word_length = 6
    alphabet_size = 4

    # SFA with ANOVA one-sided test
    window_size = 32
    p = SFA(
        word_length=word_length,
        anova=True,
        alphabet_size=alphabet_size,
        window_size=window_size,
        binning_method=binning_method,
    ).fit(X, y)

    assert p.breakpoints.shape == (word_length, alphabet_size)
    _ = p.transform(X, y)

    # SFA with first feq coefficients
    p2 = SFA(
        word_length=word_length,
        anova=False,
        alphabet_size=alphabet_size,
        window_size=window_size,
        binning_method=binning_method,
    ).fit(X, y)

    assert p.dft_length != p2.dft_length
    assert (p.breakpoints != p2.breakpoints).any()
    _ = p2.transform(X, y)


#
@pytest.mark.parametrize("word_length", [6, 7])
@pytest.mark.parametrize("alphabet_size", [4, 5])
@pytest.mark.parametrize("window_size", [5, 6])
@pytest.mark.parametrize("bigrams", [True, False])
@pytest.mark.parametrize("levels", [1, 2])
@pytest.mark.parametrize("use_fallback_dft", [True, False])
def test_word_lengths(
    word_length, alphabet_size, window_size, bigrams, levels, use_fallback_dft
):
    """Test word lengths larger than the window-length.

    Params:
    - word_length (int): The length of the words.
    - alphabet_size (int): The size of the alphabet.
    - window_size (int): The size of the sliding window.
    - bigrams (bool): Whether to use bigrams.
    - levels (int): The number of levels for multi-resolution SFA.
    - use_fallback_dft (bool): Whether to use the fallback DFT implementation.
    """
    # training data
    X = np.random.rand(10, 1, 150)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        window_size=window_size,
        bigrams=bigrams,
        levels=levels,
        use_fallback_dft=use_fallback_dft,
    ).fit(X, y)

    assert p.breakpoints is not None
    _ = p.transform(X, y)


def test_bit_size():
    """Test the bit size of transformed data by the SFA transformer."""
    # load training data
    X = np.random.rand(10, 1, 150)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    word_length = 40
    alphabet_size = 12
    window_size = 75

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        levels=2,
        bigrams=True,
        window_size=window_size,
    ).fit(X, y)

    w = p.transform(X)
    lengths = [x.bit_length() for x in list(w[0][0].keys())]

    assert np.min(lengths) > 128
    assert len(p.word_list(list(w[0][0].keys())[0])[0]) == 40


def test_typed_dict():
    """Test typed dictionaries."""
    # load training data
    X = np.random.rand(10, 1, 150)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    word_length = 6
    alphabet_size = 4

    p = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        levels=2,
        typed_dict=True,
    )
    p.fit(X, y)
    word_list = p.bag_to_string(p.transform(X, y)[0][0])

    word_length = 6
    alphabet_size = 4

    p2 = SFA(
        word_length=word_length,
        alphabet_size=alphabet_size,
        levels=2,
        typed_dict=False,
    )
    p2.fit(X, y)
    word_list2 = p2.bag_to_string(p2.transform(X, y)[0][0])

    assert word_list == word_list2


def test_sfa_fast_transform_after_fit():
    """Test transform called after fit returns the same result as fit_transform()."""
    X_train, y_train = load_unit_test(split="train")

    # Fit, then transform
    sfa = SFAFast()
    sfa.fit(X_train, y_train)
    x = sfa.transform(X_train, y_train)

    # Fit_transform, then transform
    sfa = SFAFast()
    sfa.fit_transform(X_train, y_train)
    y = sfa.transform(X_train, y_train)

    # Assert that the two csr_matrix are equal
    assert (
        x.shape == y.shape
        and x.dtype == y.dtype
        and np.all(x.indices == y.indices)
        and np.all(x.indptr == y.indptr)
        and np.allclose(x.data, y.data)
    )


def test_incorrect_paras():
    """Test incorrect parameters in SFA."""
    X, y = make_example_3d_numpy(n_cases=20, n_channels=1, n_timepoints=49)
    sfa = SFA(alphabet_size=1, word_length=0, binning_method="information-gain")
    with pytest.raises(ValueError, match="must be an integer greater than 2"):
        sfa._fit(X)
    sfa.alphabet_size = 4
    with pytest.raises(ValueError, match="must be an integer greater than 1"):
        sfa._fit(X)
    sfa.word_length = 2
    with pytest.raises(ValueError, match="Class values must be provided"):
        sfa._fit(X)
    sfa.binning_method = "Arsenal"
    with pytest.raises(TypeError, match="binning_method must be one of"):
        sfa._fit(X)
    sfa = SFA(word_length=64, alphabet_size=8)
    sfa.max_bits = 128
    sfa._typed_dict = True
    with pytest.raises(ValueError, match="Typed Dictionaries can only handle 64 bit"):
        sfa._fit(X, y)
    sfa = SFA(typed_dict=True, levels=16)
    with pytest.raises(ValueError, match="Dictionaries can only handle 15 levels"):
        sfa._fit(X, y)
    sfa = SFA(typed_dict=True, save_words=True, n_jobs=2)
    sfa.fit_transform(X, y)
    assert isinstance(sfa.words, np.ndarray)
