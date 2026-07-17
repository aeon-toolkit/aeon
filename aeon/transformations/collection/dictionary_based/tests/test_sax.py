"""Tests for the Symbolic Aggregate approXimation (SAX) transformer."""

import numpy as np
import pytest
from scipy.stats import norm

from aeon.transformations.collection.dictionary_based import SAX


def _z_normalize(X):
    """Z-normalize each channel of each series independently."""
    means = np.mean(X, axis=-1, keepdims=True)
    stds = np.std(X, axis=-1, keepdims=True)
    stds[stds == 0] = 1.0
    return (X - means) / stds


@pytest.mark.parametrize(
    "shape,n_segments,alphabet_size",
    [
        ((5, 1, 32), 4, 3),
        ((5, 1, 40), 8, 4),
        ((3, 2, 48), 6, 5),
        ((2, 4, 64), 8, 6),
    ],
)
def test_sax_standard_output_and_inverse_shapes(shape, n_segments, alphabet_size):
    """Test standard SAX and inverse SAX for several collection shapes."""
    rng = np.random.RandomState(42)
    X = _z_normalize(rng.normal(size=shape))

    sax = SAX(
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        znormalized=True,
        n_jobs=2,
    )

    X_sax = sax.fit_transform(X)
    X_inverse = sax.inverse_sax(X_sax, original_length=shape[-1])

    assert X_sax.shape == (shape[0], shape[1], n_segments)
    assert X_sax.dtype == np.dtype(np.intp)
    assert X_inverse.shape == shape
    assert np.isfinite(X_inverse).all()

    assert len(sax.breakpoints) == alphabet_size - 1
    assert len(sax.breakpoints_mid) == alphabet_size
    assert np.all(np.diff(sax.breakpoints) > 0)
    assert np.all(np.diff(sax.breakpoints_mid) > 0)


def test_sax_known_symbols_and_inverse_values():
    """Test a hand-computed SAX transformation with an exact expected word.

    The four PAA segments have means -1, -0.2, 0.2 and 1. For a standard
    Gaussian alphabet of size four, the breakpoints are approximately
    -0.674, 0 and 0.674, so the exact expected SAX word is ``a b c d``.
    """
    X = np.array(
        [[[-1.0, -1.0, -0.2, -0.2, 0.2, 0.2, 1.0, 1.0]]],
        dtype=np.float64,
    )
    alphabet = np.array(["a", "b", "c", "d"])

    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        alphabet=alphabet,
        znormalized=True,
    )

    X_sax = sax.fit_transform(X)

    expected_symbols = np.array([[["a", "b", "c", "d"]]])
    np.testing.assert_array_equal(X_sax, expected_symbols)

    X_inverse = sax.inverse_sax(X_sax, original_length=8)

    expected_midpoints = norm.ppf(np.array([1, 3, 5, 7], dtype=np.float64) / 8)
    expected_inverse = np.repeat(expected_midpoints, 2).reshape(1, 1, 8)

    np.testing.assert_allclose(
        X_inverse,
        expected_inverse,
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "alphabet",
    [
        ["a", "b", "c", "d"],
        np.array([10, 20, 30, 40]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([False, True, False, True]),
    ],
)
def test_sax_custom_alphabet_is_preserved(alphabet):
    """Test that transformed values are drawn from the custom alphabet."""
    X = np.array(
        [[[-1.0, -1.0, -0.2, -0.2, 0.2, 0.2, 1.0, 1.0]]],
        dtype=np.float64,
    )

    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        alphabet=alphabet,
        znormalized=True,
    )
    X_sax = sax.fit_transform(X)

    expected = np.asarray(alphabet)[np.array([0, 1, 2, 3])]
    np.testing.assert_array_equal(X_sax[0, 0], expected)


def test_sax_custom_alphabet_round_trip():
    """Test that a unique custom alphabet can be inverted."""
    X = np.array(
        [[[-1.0, -1.0, -0.2, -0.2, 0.2, 0.2, 1.0, 1.0]]],
        dtype=np.float64,
    )

    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        alphabet=["low", "mid-low", "mid-high", "high"],
        znormalized=True,
    )

    X_sax = sax.fit_transform(X)
    X_inverse = sax.inverse_sax(X_sax, original_length=X.shape[-1])

    assert X_sax.dtype.kind == "U"
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(
        X_inverse[0, 0],
        np.repeat(sax.breakpoints_mid, 2),
    )


def test_sax_inverse_rejects_duplicate_alphabet():
    """Test that inverse SAX rejects ambiguous duplicate symbols."""
    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        alphabet=["a", "a", "b", "c"],
        znormalized=True,
    )
    X_sax = np.array([[["a", "a", "b", "c"]]])

    with pytest.raises(ValueError, match="unique"):
        sax.inverse_sax(X_sax, original_length=8)


def test_sax_inverse_rejects_unknown_symbol():
    """Test that inverse SAX rejects symbols absent from its alphabet."""
    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=True,
    )
    X_sax = np.array([[["a", "b", "unknown", "d"]]])

    with pytest.raises(ValueError, match="Unknown SAX symbol"):
        sax.inverse_sax(X_sax, original_length=8)


def test_sax_znormalized_false_matches_manual_normalization():
    """Test that internal normalization matches manual z-normalization."""
    rng = np.random.RandomState(0)
    X = rng.normal(loc=5, scale=2, size=(5, 2, 40))
    X_normalized = _z_normalize(X)

    words_internal = SAX(
        n_segments=8,
        alphabet_size=4,
        znormalized=False,
    ).fit_transform(X)

    words_manual = SAX(
        n_segments=8,
        alphabet_size=4,
        znormalized=True,
    ).fit_transform(X_normalized)

    np.testing.assert_array_equal(words_internal, words_manual)


def test_sax_znormalized_false_handles_constant_series():
    """Test that internal normalization safely handles zero variance."""
    X = np.full((2, 1, 16), 5.0)

    X_sax = SAX(
        n_segments=4,
        alphabet_size=4,
        znormalized=False,
    ).fit_transform(X)

    # A normalized constant series is zero. np.digitize with right=False
    # places zero in bin 2 for the four-symbol Gaussian alphabet.
    np.testing.assert_array_equal(
        X_sax,
        np.full((2, 1, 4), 2, dtype=np.intp),
    )


def test_windowed_sax_normalizes_each_window_independently():
    """Test that windowed SAX z-normalizes each window independently."""
    X = np.array(
        [
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    100.0,
                    100.0,
                    110.0,
                    110.0,
                ]
            ]
        ],
        dtype=np.float64,
    )

    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=False,
        window_size=4,
        stride=4,
    )

    X_sax = sax.fit_transform(X)
    expected = np.array([[[["a", "d"], ["a", "d"]]]])
    np.testing.assert_array_equal(X_sax, expected)


def test_windowed_sax_matches_manual_per_window_normalization():
    """Test windowed normalization against manually normalized windows."""
    X = np.array(
        [
            [
                [
                    1.0,
                    2.0,
                    4.0,
                    8.0,
                    10.0,
                    11.0,
                    13.0,
                    20.0,
                ]
            ]
        ],
        dtype=np.float64,
    )

    window_size = 4
    stride = 4

    words_internal = SAX(
        n_segments=2,
        alphabet_size=4,
        znormalized=False,
        window_size=window_size,
        stride=stride,
    ).fit_transform(X)

    windows = X.reshape(1, 1, 2, window_size)
    means = windows.mean(axis=-1, keepdims=True)
    stds = windows.std(axis=-1, keepdims=True)
    stds[stds == 0] = 1.0
    windows_normalized = (windows - means) / stds
    X_manual = windows_normalized.reshape(2, 1, window_size)

    words_manual = SAX(
        n_segments=2,
        alphabet_size=4,
        znormalized=True,
    ).fit_transform(X_manual)
    words_manual = words_manual.reshape(1, 2, 1, 2).transpose(0, 2, 1, 3)

    np.testing.assert_array_equal(words_internal, words_manual)


def test_windowed_sax_differs_from_whole_series_normalization():
    """Test that per-window and whole-series normalization are distinct."""
    X = np.array(
        [
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    100.0,
                    100.0,
                    110.0,
                    110.0,
                ]
            ]
        ],
        dtype=np.float64,
    )

    words_per_window = SAX(
        n_segments=2,
        alphabet_size=4,
        znormalized=False,
        window_size=4,
        stride=4,
    ).fit_transform(X)

    X_whole_normalized = _z_normalize(X)
    words_whole_normalized = SAX(
        n_segments=2,
        alphabet_size=4,
        znormalized=True,
        window_size=4,
        stride=4,
    ).fit_transform(X_whole_normalized)

    assert not np.array_equal(words_per_window, words_whole_normalized)


@pytest.mark.parametrize(
    "window_size,stride,expected_n_windows",
    [
        (8, 8, 4),
        (8, 4, 7),
        (8, 2, 13),
        (16, 8, 3),
        (32, 1, 1),
    ],
)
def test_windowed_sax_output_shape(window_size, stride, expected_n_windows):
    """Test window counts and output shape for several window setups."""
    rng = np.random.RandomState(7)
    X = _z_normalize(rng.normal(size=(3, 2, 32)))

    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        znormalized=True,
        window_size=window_size,
        stride=stride,
    )
    X_sax = sax.fit_transform(X)

    assert X_sax.shape == (
        X.shape[0],
        X.shape[1],
        expected_n_windows,
        sax.n_segments,
    )


def test_windowed_sax_known_words_non_overlapping():
    """Test exact words after independent per-window normalization.

    The two windows differ only in scale. After each window is independently
    z-normalized, both become ``[-1, -1, 1, 1]``. Their two PAA segments are
    therefore -1 and 1, yielding the SAX word ``a d`` for both windows.
    """
    X = np.array(
        [[[-1.0, -1.0, 1.0, 1.0, -0.2, -0.2, 0.2, 0.2]]],
        dtype=np.float64,
    )

    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=False,
        window_size=4,
        stride=4,
    )

    X_sax = sax.fit_transform(X)

    expected = np.array([[[["a", "d"], ["a", "d"]]]])
    np.testing.assert_array_equal(X_sax, expected)


def test_windowed_sax_known_words_without_internal_normalization():
    """Test exact words when input windows are treated as pre-normalized."""
    X = np.array(
        [[[-1.0, -1.0, 1.0, 1.0, -0.2, -0.2, 0.2, 0.2]]],
        dtype=np.float64,
    )

    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=True,
        window_size=4,
        stride=4,
    )

    X_sax = sax.fit_transform(X)

    expected = np.array([[[["a", "d"], ["b", "c"]]]])
    np.testing.assert_array_equal(X_sax, expected)


def test_windowed_inverse_known_non_overlapping_result():
    """Test exact reconstruction of known non-overlapping SAX words."""
    X_sax = np.array([[[["a", "d"], ["b", "c"]]]])

    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=True,
        window_size=4,
        stride=4,
    )

    X_inverse = sax.inverse_sax(X_sax, original_length=8)

    mids = sax.breakpoints_mid
    expected = np.array(
        [
            [
                [
                    mids[0],
                    mids[0],
                    mids[3],
                    mids[3],
                    mids[1],
                    mids[1],
                    mids[2],
                    mids[2],
                ]
            ]
        ]
    )

    np.testing.assert_allclose(X_inverse, expected, rtol=0, atol=1e-12)


def test_windowed_inverse_averages_overlaps():
    """Test overlap-add reconstruction by averaging overlapping windows."""
    # Window 0 reconstructs [m0, m0, m3, m3].
    # Window 1 starts at t=2 and reconstructs [m1, m1, m2, m2].
    X_sax = np.array([[[["a", "d"], ["b", "c"]]]])

    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=True,
        window_size=4,
        stride=2,
    )

    X_inverse = sax.inverse_sax(X_sax, original_length=6)

    m0, m1, m2, m3 = sax.breakpoints_mid
    expected = np.array(
        [
            [
                [
                    m0,
                    m0,
                    (m3 + m1) / 2,
                    (m3 + m1) / 2,
                    m2,
                    m2,
                ]
            ]
        ]
    )

    np.testing.assert_allclose(X_inverse, expected, rtol=0, atol=1e-12)


def test_windowed_inverse_infers_covered_length():
    """Test that windowed inverse can infer its covered output length."""
    X_sax = np.array([[[["a", "d"], ["b", "c"], ["c", "b"]]]])

    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        alphabet=["a", "b", "c", "d"],
        znormalized=True,
        window_size=4,
        stride=2,
    )

    X_inverse = sax.inverse_sax(X_sax)

    expected_length = (X_sax.shape[2] - 1) * sax.stride + sax.window_size
    assert X_inverse.shape == (1, 1, expected_length)
    assert np.isfinite(X_inverse).all()


def test_windowed_sax_multivariate_custom_alphabet():
    """Test windowed SAX with multiple channels and a string alphabet."""
    rng = np.random.RandomState(11)
    X = _z_normalize(rng.normal(size=(2, 3, 24)))

    alphabet = np.array(["A", "B", "C", "D", "E"])
    sax = SAX(
        n_segments=4,
        alphabet_size=5,
        alphabet=alphabet,
        znormalized=True,
        window_size=12,
        stride=6,
    )

    X_sax = sax.fit_transform(X)
    X_inverse = sax.inverse_sax(X_sax, original_length=X.shape[-1])

    assert X_sax.shape == (2, 3, 3, 4)
    assert X_inverse.shape == X.shape
    assert set(np.unique(X_sax)).issubset(set(alphabet))


@pytest.mark.parametrize(
    "window_size,stride,error_type,match",
    [
        (0, 1, ValueError, "greater than 0"),
        (-1, 1, ValueError, "greater than 0"),
        (3, 1, ValueError, "at least as large as n_segments"),
        (33, 1, ValueError, "greater than the series length"),
        (8, 0, ValueError, "positive integer"),
        (8, -1, ValueError, "positive integer"),
    ],
)
def test_invalid_window_parameters_raise(window_size, stride, error_type, match):
    """Test validation of invalid window configurations."""
    X = np.zeros((1, 1, 32), dtype=np.float64)

    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        znormalized=True,
        window_size=window_size,
        stride=stride,
    )

    with pytest.raises(error_type, match=match):
        sax.fit_transform(X)


def test_non_integer_window_size_raises():
    """Test that a non-integer window size is rejected."""
    X = np.zeros((1, 1, 32), dtype=np.float64)
    sax = SAX(
        n_segments=4,
        alphabet_size=4,
        znormalized=True,
        window_size=8.5,
        stride=1,
    )

    with pytest.raises(TypeError, match="integer"):
        sax.fit_transform(X)


def test_standard_inverse_requires_original_length():
    """Test that standard inverse SAX requires a target length."""
    sax = SAX(n_segments=4, alphabet_size=4, znormalized=True)
    X_sax = np.zeros((1, 1, 4), dtype=np.intp)

    with pytest.raises(ValueError, match="original_length"):
        sax.inverse_sax(X_sax)


def test_inverse_rejects_invalid_number_of_dimensions():
    """Test that inverse SAX rejects arrays that are not 3D or 4D."""
    sax = SAX(n_segments=4, alphabet_size=4, znormalized=True)

    with pytest.raises(ValueError, match="shape"):
        sax.inverse_sax(np.zeros((1, 4), dtype=np.intp), original_length=8)


def test_windowed_inverse_rejects_too_short_original_length():
    """Test that original_length cannot truncate covered windows."""
    X_sax = np.zeros((1, 1, 3, 2), dtype=np.intp)
    sax = SAX(
        n_segments=2,
        alphabet_size=4,
        znormalized=True,
        window_size=4,
        stride=2,
    )

    # Three windows cover (3 - 1) * 2 + 4 = 8 points.
    with pytest.raises(ValueError, match="smaller"):
        sax.inverse_sax(X_sax, original_length=7)


def test_unsupported_distribution_raises():
    """Test that unsupported distributions are rejected."""
    with pytest.raises(NotImplementedError):
        SAX(distribution="bogus", znormalized=True)


def test_alphabet_length_must_match_alphabet_size():
    """Test that alphabet length matches alphabet_size."""
    with pytest.raises((AssertionError, ValueError)):
        SAX(
            alphabet_size=4,
            alphabet=["a", "b", "c"],
            znormalized=True,
        )


def test_sax_default_znormalized_is_true():
    """Test that SAX treats input as pre-normalized by default."""
    assert SAX().znormalized is True


def test_sax_get_test_params():
    """Test that the estimator's default test parameters are usable."""
    params = SAX._get_test_params()
    sax = SAX(**params)

    assert isinstance(sax, SAX)
