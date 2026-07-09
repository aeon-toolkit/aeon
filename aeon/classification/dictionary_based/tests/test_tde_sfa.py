"""Tests for the TDE-specific SFA implementation.

These tests check that the private SFA rewrite keeps the TDE-facing behaviour
of generic SFA while covering the specialised array-backed bag and numba helper
paths used by ``IndividualTDE``.
"""

import math

import numpy as np
import pytest

from aeon.classification.dictionary_based._tde_sfa import (
    _DBL_MAX,
    _TDESFA,
    _bags_from_dft,
    _best_split,
    _binning_dft_all,
    _entropy,
    _histogram_intersection,
    _igb_all,
    _incremental_stds,
    _mft_all,
    combine_dim_bags,
    loocv_train_acc,
    nn_predict_loocv,
    nn_similarities_all,
)
from aeon.transformations.collection.dictionary_based import SFA as OldSFA


def _make_toy_data(n_cases=9, n_timepoints=24, seed=0):
    """Make deterministic univariate data with class-dependent shape."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_cases, 1, n_timepoints), dtype=np.float64)
    y = np.arange(n_cases) % 3
    t = np.linspace(0, 2 * np.pi, n_timepoints)

    for i, cls in enumerate(y):
        X[i, 0] = np.sin((cls + 1) * t) + 0.05 * rng.normal(size=n_timepoints)
        X[i, 0] += cls * 0.2

    return X, y


def _old_sfa(word_length, window_size, norm, levels, binning_method, bigrams):
    """Construct the generic SFA with the options used by TDE."""
    return OldSFA(
        word_length=word_length,
        alphabet_size=4,
        window_size=window_size,
        norm=norm,
        levels=levels,
        binning_method=binning_method,
        bigrams=bigrams,
        remove_repeat_words=True,
        lower_bounding=False,
        save_words=False,
        use_fallback_dft=True,
        typed_dict=True,
        n_jobs=1,
    )


def _convert_old_bags(bags, levels=1, word_length=8):
    """Convert generic SFA dict bags to sorted ``(key1, key2, count)`` triples."""
    level_bits = 0
    max_unigram = 1 << (word_length * 2)
    if levels > 1:
        level_bits = math.ceil(math.log2(sum(2**i for i in range(levels))))

    out = []
    for bag in bags:
        rows = []
        for key, value in bag.items():
            if isinstance(key, tuple):
                rows.append((int(key[0]), int(key[1]), int(value)))
            elif levels > 1:
                word = int(key) >> level_bits
                quadrant = int(key) & ((1 << level_bits) - 1)
                if word >= max_unigram:
                    quadrant = -1
                rows.append((word, quadrant, int(value)))
            else:
                rows.append((int(key), 0, int(value)))
        out.append(sorted(rows))
    return out


def _convert_new_bags(bags):
    """Convert array-backed TDE SFA bags to sorted per-case triples."""
    keys1, keys2, counts, offsets = bags
    out = []
    for i in range(len(offsets) - 1):
        out.append(
            [
                (int(keys1[j]), int(keys2[j]), int(counts[j]))
                for j in range(offsets[i], offsets[i + 1])
            ]
        )
    return out


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"word_length": 0}, "word_length"),
        ({"levels": 0}, "levels"),
        ({"levels": 4}, "levels"),
        ({"binning_method": "bad"}, "binning_method"),
        ({"word_length": 33}, "64 bits"),
        ({"word_length": 17, "bigrams": True}, "64 bits"),
    ],
)
def test_tde_sfa_rejects_parameters_outside_tde_space(kwargs, match):
    """Test constructor validation for options unsupported by TDE.

    The private implementation only supports TDE's fixed alphabet, pyramid
    depth and key-width assumptions, so invalid parameters should fail at
    construction rather than creating an object that later mis-encodes bags.
    """
    with pytest.raises(ValueError, match=match):
        _TDESFA(**kwargs)


def test_tde_sfa_validates_input_shape_and_fit_requirements():
    """Test runtime validation for input shape and fit prerequisites.

    TDE slices one channel at a time before calling ``_TDESFA``. This checks
    that accidental multivariate input, oversized windows, missing IGB labels
    and missing retained binning DFTs all raise clear errors.
    """
    X, _ = _make_toy_data(n_cases=4, n_timepoints=12)

    with pytest.raises(ValueError, match="univariate"):
        _TDESFA()._check_X(np.zeros((2, 2, 12)))

    with pytest.raises(ValueError, match="larger"):
        _TDESFA(window_size=13).fit(X)

    with pytest.raises(ValueError, match="y is required"):
        _TDESFA(binning_method="information-gain").fit(X)

    with pytest.raises(ValueError, match="keep_binning_dft"):
        _TDESFA().binning_bags()


@pytest.mark.parametrize(
    "word_length, window_size, norm, levels, binning_method, bigrams",
    [
        (6, 8, False, 1, "equi-depth", False),
        (8, 8, False, 1, "equi-depth", True),
        (6, 8, True, 2, "equi-depth", True),
        (6, 8, False, 3, "information-gain", False),
    ],
)
def test_tde_sfa_matches_generic_sfa_for_tde_configurations(
    word_length, window_size, norm, levels, binning_method, bigrams
):
    """Test bag parity with generic SFA across representative TDE settings.

    The configurations cover flat bags, bigrams, normalised transforms,
    spatial-pyramid bags and information-gain binning. Matching generic SFA
    here is the main behavioural contract for replacing TDE's transformer.
    """
    X, y = _make_toy_data()

    old = _old_sfa(word_length, window_size, norm, levels, binning_method, bigrams)
    old.fit(X, y)
    expected = _convert_old_bags(old.transform(X)[0], levels, word_length)

    new = _TDESFA(
        word_length=word_length,
        window_size=window_size,
        norm=norm,
        levels=levels,
        binning_method=binning_method,
        bigrams=bigrams,
    )
    actual = _convert_new_bags(new.fit_transform(X, y))

    assert actual == expected


def test_transform_handles_fit_array_and_new_array_paths():
    """Test both transform paths: cached fit MFT and new-data MFT.

    ``fit_transform`` stores the MFT computed during fitting. A subsequent
    ``transform`` on the same checked array should reuse it, while transform
    on different data should compute a fresh MFT and still return valid bags.
    """
    X, y = _make_toy_data(n_cases=6, n_timepoints=20, seed=1)
    X_2d = np.ascontiguousarray(X[:, 0])
    X_new, _ = _make_toy_data(n_cases=6, n_timepoints=20, seed=2)

    sfa = _TDESFA(word_length=4, window_size=6, bigrams=True)
    fit_bags = _convert_new_bags(sfa.fit_transform(X_2d, y))

    assert _convert_new_bags(sfa.transform(X_2d)) == fit_bags
    assert len(_convert_new_bags(sfa.transform(X_new))) == X_new.shape[0]


def test_binning_bags_use_retained_direct_dft():
    """Test reduced binning bags used by multivariate dimension selection.

    When ``keep_binning_dft=True``, ``binning_bags`` should build bags from
    the retained direct DFT rows, matching generic SFA's supplied-DFT path.
    """
    X, y = _make_toy_data(n_cases=6, n_timepoints=20)
    old = _old_sfa(4, 6, False, 2, "equi-depth", False)
    old.keep_binning_dft = True
    old.fit(X, y)
    expected = _convert_old_bags(old.transform(X)[0], levels=2, word_length=4)

    new = _TDESFA(
        word_length=4,
        window_size=6,
        levels=2,
        keep_binning_dft=True,
    )
    new.fit(X, y)

    assert _convert_new_bags(new.binning_bags()) == expected


def test_mcb_breakpoints_use_two_decimal_equi_depth_bins():
    """Test MCB equi-depth breakpoints and two-decimal rounding.

    Generic SFA rounds Fourier values to two decimals before choosing
    equi-depth cut points. This asserts the same cut points and the sentinel
    final breakpoint used by the fixed alphabet.
    """
    dft = np.array(
        [
            [0.111, 9.0],
            [0.226, 8.0],
            [0.333, 7.0],
            [0.449, 6.0],
            [0.551, 5.0],
            [0.662, 4.0],
            [0.778, 3.0],
            [0.889, 2.0],
        ]
    )
    breakpoints = _TDESFA(word_length=2)._mcb_equi_depth(dft)

    np.testing.assert_array_equal(breakpoints[0, :3], np.array([0.33, 0.55, 0.78]))
    assert breakpoints[0, 3] == _DBL_MAX
    np.testing.assert_array_equal(breakpoints[1, :3], np.array([4.0, 6.0, 8.0]))


def test_information_gain_breakpoints_handle_split_and_no_split_cases():
    """Test IGB breakpoint output for splittable and constant letters.

    One letter has class-separating values and should receive finite sorted
    thresholds; the constant letter has no useful split and should remain at
    the maximum-value sentinel breakpoints.
    """
    dft = np.array(
        [
            [0.0, 1.0],
            [0.1, 1.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 1.0],
            [2.1, 1.0],
        ]
    )
    y = np.array(["a", "a", "b", "b", "c", "c"])
    breakpoints = _TDESFA(word_length=2, binning_method="information-gain")._igb(dft, y)

    assert np.isfinite(breakpoints[0, 0])
    assert np.all(breakpoints[0] == np.sort(breakpoints[0]))
    assert np.all(breakpoints[1] == _DBL_MAX)


def test_incremental_stds_use_one_for_constant_windows():
    """Test sliding-window standard deviations guard constant windows.

    Constant subsequences would otherwise divide by zero during normalisation;
    the helper should replace near-zero standard deviations with one.
    """
    stds = _incremental_stds(np.ones(6), 3, 4)

    np.testing.assert_array_equal(stds, np.ones(3))


def test_mft_and_direct_binning_dft_match_manual_first_window():
    """Test the first MFT window against the direct DFT binning path.

    The first sliding MFT window is computed directly before incremental
    updates start. This checks that it agrees with the direct DFT helper used
    for binning rows.
    """
    X = np.array([[1.0, 2.0, 4.0, 8.0, 16.0]], dtype=np.float64)
    window_size = 4
    inverse_sqrt = 1.0 / math.sqrt(window_size)

    mft = _mft_all(X, window_size, 4, 0, inverse_sqrt)
    direct = _binning_dft_all(X, window_size, 4, False, inverse_sqrt, 2)

    np.testing.assert_allclose(mft[0, 0], direct[0, 0])


def test_normed_binning_dft_drops_first_coefficient_pair():
    """Test normalised direct DFT skips the leading coefficient pair.

    With ``norm=True``, SFA drops the mean coefficient pair. This checks the
    direct binning DFT returns the expected output shape and non-empty values
    after applying that offset.
    """
    X = np.array([[1.0, 2.0, 4.0, 8.0, 16.0]], dtype=np.float64)
    out = _binning_dft_all(X, 4, 4, True, 1.0 / math.sqrt(4), 2)

    assert out.shape == (1, 2, 4)
    assert not np.allclose(out[0, 0], 0)


def test_direct_binning_dft_uses_unit_std_for_constant_windows():
    """Test direct DFT normalisation also protects constant windows.

    This is the non-sliding DFT path used for IGB and retained binning bags;
    constant input should produce finite coefficients rather than divide by
    zero.
    """
    X = np.ones((1, 5), dtype=np.float64)
    out = _binning_dft_all(X, 4, 4, False, 1.0 / math.sqrt(4), 2)

    assert np.all(np.isfinite(out))


def test_bags_from_dft_pack_flat_unigrams_and_bigrams():
    """Test flat bag encoding for unigrams and bigrams.

    With ``levels=1``, unigrams and packed bigrams intentionally share the
    same key space and have ``key2 == 0``. The fixture includes repeated and
    alternating words to check aggregation in that representation.
    """
    dfts = np.array(
        [
            [[0.0], [2.0], [0.0], [2.0]],
            [[2.0], [2.0], [0.0], [0.0]],
        ],
        dtype=np.float64,
    )
    breakpoints = np.array([[0.5, 1.5, 2.5, _DBL_MAX]])

    bags = _bags_from_dft(dfts, breakpoints, 1, 2, 1, 4, 1, True)

    assert _convert_new_bags(bags) == [
        [(0, 0, 2), (2, 0, 4), (8, 0, 1)],
        [(0, 0, 2), (2, 0, 1), (8, 0, 1), (10, 0, 1)],
    ]


def test_bags_from_dft_pack_pyramid_unigrams_and_bigrams():
    """Test spatial-pyramid bag encoding and weights.

    For ``levels > 1``, unigrams carry a quadrant in ``key2`` and are weighted
    by pyramid level, while bigrams are stored separately with ``key2 == -1``.
    """
    dfts = np.array([[[0.0], [2.0], [0.0], [2.0]]], dtype=np.float64)
    breakpoints = np.array([[0.5, 1.5, 2.5, _DBL_MAX]])

    bags = _bags_from_dft(dfts, breakpoints, 1, 2, 1, 4, 3, True)

    assert _convert_new_bags(bags) == [
        [
            (0, 0, 2),
            (0, 1, 2),
            (0, 2, 2),
            (0, 3, 4),
            (0, 5, 4),
            (2, -1, 2),
            (2, 0, 2),
            (2, 1, 2),
            (2, 2, 2),
            (2, 4, 4),
            (2, 6, 4),
            (8, -1, 1),
        ]
    ]


def test_bags_from_dft_applies_numerosity_reduction():
    """Test numerosity reduction before spatial-pyramid aggregation.

    Adjacent repeated words should collapse to a single word event. The
    expected bag proves repeats are removed before level weights are applied.
    """
    dfts = np.zeros((1, 4, 1), dtype=np.float64)
    breakpoints = np.array([[0.5, 1.5, 2.5, _DBL_MAX]])

    bags = _bags_from_dft(dfts, breakpoints, 1, 2, 2, 8, 2, False)

    assert _convert_new_bags(bags) == [[(0, 0, 1), (0, 1, 2)]]


def test_entropy_and_best_split_cover_unsplittable_segments():
    """Test entropy and split helpers on cases with no valid split.

    A pure node and a one-sample segment should report no candidate split,
    which is required for IGB tree growth to stop cleanly.
    """
    assert _entropy(np.array([2.0, 0.0]), 2) == 0.0

    xs = np.array([1.0])
    ys = np.array([0], dtype=np.int64)
    assert _best_split(xs, ys, 0, 1, 2, 1) == (-1.0, -1, 0.0)

    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([0, 0, 0], dtype=np.int64)
    assert _best_split(xs, ys, 0, 3, 1, 3) == (-1.0, -1, 0.0)


def test_best_split_returns_midpoint_guarded_threshold():
    """Test the midpoint guard used by IGB split thresholds.

    When adjacent floating-point values are extremely close, their midpoint
    can round to the right-hand value. The helper should move the threshold
    back to the left value to preserve sklearn-compatible split semantics.
    """
    left = np.nextafter(1.0, 0.0)
    right = 1.0
    xs = np.array([left, right])
    ys = np.array([0, 1], dtype=np.int64)

    gain, pos, threshold = _best_split(xs, ys, 0, 2, 2, 2)

    assert gain > 0
    assert pos == 0
    assert threshold == left


def test_igb_all_stops_when_no_candidate_splits_exist():
    """Test IGB tree growth stops for constant features.

    Constant columns have no valid threshold candidates. The helper should
    return zero thresholds for every letter rather than fabricating splits.
    """
    dft = np.ones((4, 2), dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)

    thresholds, n_thresholds = _igb_all(dft, y, 2)

    np.testing.assert_array_equal(thresholds, np.zeros((2, 3)))
    np.testing.assert_array_equal(n_thresholds, np.zeros(2, dtype=np.int64))


def test_histogram_intersection_and_nearest_neighbour_helpers():
    """Test sorted merge similarity and leave-one-out nearest neighbour.

    The key arrays include matching keys and keys present on only one side, so
    the test covers equality, left-advance and right-advance branches before
    checking the nearest-neighbour index chosen from the resulting scores.
    """
    keys1 = np.array([1, 3, 5, 1, 3, 6, 2, 5], dtype=np.int64)
    keys2 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    counts = np.array([2, 1, 4, 1, 5, 1, 3, 2], dtype=np.uint32)
    offsets = np.array([0, 3, 6, 8], dtype=np.int64)

    assert _histogram_intersection(keys1, keys2, counts, 0, 3, 3, 6) == 2
    assert _histogram_intersection(keys1, keys2, counts, 3, 6, 6, 8) == 0
    assert nn_predict_loocv(keys1, keys2, counts, offsets, 0) == 1


def test_nn_similarities_all_scores_every_test_train_pair():
    """Test the batched test-to-train similarity matrix used by predict.

    ``nn_similarities_all`` scores every test bag against every train bag in
    one call. The fixture includes matching keys, keys on only one side and
    an empty overlap, checking all merge branches and the matrix layout.
    """
    train_keys1 = np.array([1, 3, 5, 2, 5], dtype=np.int64)
    train_keys2 = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    train_counts = np.array([2, 1, 4, 3, 2], dtype=np.uint32)
    train_offsets = np.array([0, 3, 5], dtype=np.int64)

    test_keys1 = np.array([1, 4, 5, 9], dtype=np.int64)
    test_keys2 = np.array([0, 0, 0, 0], dtype=np.int64)
    test_counts = np.array([1, 7, 1, 2], dtype=np.uint32)
    test_offsets = np.array([0, 3, 4], dtype=np.int64)

    sims = nn_similarities_all(
        train_keys1,
        train_keys2,
        train_counts,
        train_offsets,
        test_keys1,
        test_keys2,
        test_counts,
        test_offsets,
    )

    # test bag 0 = {1:1, 4:7, 5:1}: min counts against {1:2, 3:1, 5:4} and
    # {2:3, 5:2}; test bag 1 = {9:2} shares no keys with either train bag
    np.testing.assert_array_equal(sims, np.array([[2, 1], [0, 0]]))


def test_loocv_train_acc_predictions_and_early_abandon():
    """Test the symmetric LOOCV kernel used for member accuracy estimates.

    Bags 0 and 1 are near-identical and share a class, bag 2 is distinct, so
    every leave-one-out prediction is derivable by hand. With an unreachable
    required_correct the kernel must stop before completing all rows and
    report the abandon sentinel instead of an accuracy count.
    """
    # bag 0 = {1:2, 3:1}, bag 1 = {1:2, 3:2}, bag 2 = {9:5}
    keys1 = np.array([1, 3, 1, 3, 9], dtype=np.int64)
    keys2 = np.zeros(5, dtype=np.int64)
    counts = np.array([2, 1, 2, 2, 5], dtype=np.uint32)
    offsets = np.array([0, 2, 4, 5], dtype=np.int64)
    y_codes = np.array([0, 0, 1], dtype=np.int64)

    n_done, correct, preds = loocv_train_acc(keys1, keys2, counts, offsets, y_codes, 0)

    assert n_done == 3
    # bags 0 and 1 pick each other (correct); bag 2 shares no words with
    # either neighbour, ties at similarity 0 and takes the first, bag 0
    np.testing.assert_array_equal(preds, np.array([1, 0, 0]))
    assert correct == 2

    # with these labels bag 0 is mispredicted immediately, making three
    # correct predictions unreachable, so the pass abandons before row 1
    y_wrong = np.array([1, 0, 0], dtype=np.int64)
    n_done, correct, _ = loocv_train_acc(keys1, keys2, counts, offsets, y_wrong, 3)

    assert correct == -1
    assert n_done == 1


def test_combine_dim_bags_merges_sorted_per_dimension_streams():
    """Test the k-way merge that builds multivariate bags.

    Two single-case dimension bags are merged for levels > 1: unigram key2
    values are shifted and tagged with their dimension, the bigram key2 of -1
    keeps its negative tag, and the output must be sorted with all counts
    preserved (no keys from different dimensions may combine).
    """
    # dim 3: keys (5, -1) bigram and (5, 0) unigram; dim 6: key (5, 1)
    all_k1 = np.array([5, 5, 5], dtype=np.int64)
    all_k2 = np.array([-1, 0, 1], dtype=np.int64)
    all_v = np.array([2, 3, 4], dtype=np.uint32)
    dim_case_offsets = np.array([[0, 2], [0, 1]], dtype=np.int64)
    dim_starts = np.array([0, 2], dtype=np.int64)
    dims = np.array([3, 6], dtype=np.int64)
    highest_dim_bit = 3

    keys1, keys2, counts, offsets = combine_dim_bags(
        all_k1, all_k2, all_v, dim_case_offsets, dim_starts, dims, 2, highest_dim_bit
    )

    np.testing.assert_array_equal(offsets, np.array([0, 3]))
    np.testing.assert_array_equal(keys1, np.array([5, 5, 5]))
    # (-1 << 3) | 3 = -5 sorts first, then (0 << 3) | 3 = 3, then
    # (1 << 3) | 6 = 14
    np.testing.assert_array_equal(keys2, np.array([-5, 3, 14]))
    np.testing.assert_array_equal(counts, np.array([2, 3, 4]))


def test_combine_dim_bags_uses_dimension_as_key2_for_flat_bags():
    """Test the k-way merge key encoding when levels == 1.

    Flat bags store the dimension itself in key2, so equal words from
    different dimensions must stay separate rows, interleaved in sorted
    (key1, key2) order.
    """
    all_k1 = np.array([5, 7, 5], dtype=np.int64)
    all_k2 = np.zeros(3, dtype=np.int64)
    all_v = np.array([1, 2, 3], dtype=np.uint32)
    dim_case_offsets = np.array([[0, 2], [0, 1]], dtype=np.int64)
    dim_starts = np.array([0, 2], dtype=np.int64)
    dims = np.array([0, 4], dtype=np.int64)

    keys1, keys2, counts, offsets = combine_dim_bags(
        all_k1, all_k2, all_v, dim_case_offsets, dim_starts, dims, 1, 3
    )

    np.testing.assert_array_equal(keys1, np.array([5, 5, 7]))
    np.testing.assert_array_equal(keys2, np.array([0, 4, 0]))
    np.testing.assert_array_equal(counts, np.array([1, 3, 2]))
    np.testing.assert_array_equal(offsets, np.array([0, 3]))
