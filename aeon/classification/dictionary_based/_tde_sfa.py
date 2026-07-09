"""Lean SFA reimplementation specialised for TDE.

A from-scratch rewrite of the SFA transform covering exactly what TDE uses,
nothing else:

- alphabet size is fixed at 4 (2 bits per letter, branchless letter lookup)
- numerosity reduction (remove_repeat_words) is always on
- univariate 2D input (n_cases, n_timepoints); TDE slices channels
- strictly sequential: no joblib, no prange, no threading of any kind
- bags are array-native: each case's bag is a run of rows in concatenated
  (key1, key2, count) arrays, sorted lexicographically by key, ready for
  merge-based histogram intersection. No dicts anywhere.
- the sliding-window MFT is computed once in fit and reused by transform on
  the same data; MCB breakpoints use rows from the MFT, while IGB and reduced
  dimension-selection bags use the old direct non-overlapping DFT path because
  tiny MFT/direct roundoff differences can change IGB thresholds
- IGB (information gain binning) is implemented directly in numba as a
  best-first entropy tree on one feature, replacing per-letter sklearn
  DecisionTreeClassifier fits

Key encoding (matches the aeon typed-dict SFA logically):
- levels == 1: key1 = word, or (prev_word << word_bits) | word for bigrams;
  key2 = 0 always
- levels > 1: key1 = word (or packed bigram); key2 = quadrant for unigrams,
  -1 for bigrams

Remaining options are the TDE parameter space: word_length, window_size,
norm, levels, MCB equi-depth or IGB binning, bigrams, and binning_bags()
for multivariate dimension selection.
"""

import math
import sys

import numpy as np
from numba import njit

_DBL_MAX = sys.float_info.max

ALPHABET_SIZE = 4
LETTER_BITS = 2


class _TDESFA:
    """Symbolic Fourier Approximation transform, TDE feature set only.

    Parameters
    ----------
    word_length : int, default=8
        Number of Fourier coefficients (letters) per word.
    window_size : int, default=12
        Length of the sliding window.
    norm : bool, default=False
        If True, drop the first Fourier coefficient pair (mean normalise).
    levels : int, default=1
        Number of spatial pyramid levels.
    binning_method : str, default="equi-depth"
        "equi-depth" (MCB) or "information-gain" (IGB).
    bigrams : bool, default=False
        Whether to add bigram words (pairs of words window_size apart).
    keep_binning_dft : bool, default=False
        Keep the binning DFT after fit so `binning_bags()` can build the
        reduced bags used by TDE's multivariate dimension selection.

    Attributes
    ----------
    breakpoints : 2D np.ndarray (word_length, 4)
        Discretisation boundaries per letter, last column is DBL_MAX.
    """

    def __init__(
        self,
        word_length=8,
        window_size=12,
        norm=False,
        levels=1,
        binning_method="equi-depth",
        bigrams=False,
        keep_binning_dft=False,
    ):
        if word_length < 1:
            raise ValueError("word_length must be at least 1")
        if not 1 <= levels <= 3:
            raise ValueError("levels must be 1, 2 or 3 (the TDE parameter space)")
        if binning_method not in ("equi-depth", "information-gain"):
            raise ValueError(
                "binning_method must be 'equi-depth' or 'information-gain'"
            )

        self.word_length = word_length
        self.window_size = window_size
        self.norm = norm
        self.levels = levels
        self.binning_method = binning_method
        self.bigrams = bigrams
        self.keep_binning_dft = keep_binning_dft

        self.word_bits = word_length * LETTER_BITS
        if self.word_bits * (2 if bigrams else 1) > 64:
            raise ValueError("words (and bigrams) must fit in 64 bits")

        # number of Fourier values kept per window (even, pairs of real/imag)
        self.dft_length = word_length + word_length % 2
        # extra leading pair computed then dropped when norm is used
        self.norm_offset = 2 if norm else 0
        self.inverse_sqrt_win_size = 1.0 / math.sqrt(window_size)

        self.breakpoints = None
        self.n_timepoints = 0
        self._fit_X = None
        self._fit_mft = None
        self._binning_dft = None

    def fit(self, X, y=None):
        """Learn breakpoints from X (2D or squeezable 3D array)."""
        X = self._check_X(X)
        n_cases, self.n_timepoints = X.shape
        if self.window_size > self.n_timepoints:
            raise ValueError("window_size larger than series length")
        if self.binning_method == "information-gain" and y is None:
            raise ValueError("y is required for information gain binning")

        mft = _mft_all(
            X,
            self.window_size,
            self.dft_length + self.norm_offset,
            self.norm_offset,
            self.inverse_sqrt_win_size,
        )

        num_windows_per_inst = int(math.ceil(self.n_timepoints / self.window_size))
        idx = np.empty(num_windows_per_inst, dtype=np.int64)
        for i in range(num_windows_per_inst - 1):
            idx[i] = i * self.window_size
        idx[-1] = self.n_timepoints - self.window_size

        direct_binning_dft = None
        if self.binning_method == "information-gain" or self.keep_binning_dft:
            direct_binning_dft = _binning_dft_all(
                X,
                self.window_size,
                self.dft_length,
                self.norm,
                self.inverse_sqrt_win_size,
                num_windows_per_inst,
            )

        binning_dft = (
            direct_binning_dft
            if self.binning_method == "information-gain"
            else mft[:, idx, :]
        )

        flat = np.ascontiguousarray(
            binning_dft.reshape(n_cases * num_windows_per_inst, mft.shape[2])
        )
        if self.binning_method == "information-gain":
            # one label per binning window
            self.breakpoints = self._igb(flat, np.repeat(y, num_windows_per_inst))
        else:
            self.breakpoints = self._mcb_equi_depth(flat)

        self._fit_X = X
        self._fit_mft = mft
        self._binning_dft = (
            np.ascontiguousarray(direct_binning_dft) if self.keep_binning_dft else None
        )
        return self

    def transform(self, X):
        """Transform X into bags of words.

        Returns
        -------
        (keys1, keys2, counts, offsets) :
            Concatenated per-case bags. Case i's bag is rows
            offsets[i]:offsets[i+1], sorted lexicographically by
            (keys1, keys2). counts is uint32.
        """
        X = self._check_X(X)
        if self._fit_X is not None and X is self._fit_X:
            mft = self._fit_mft
        else:
            mft = _mft_all(
                X,
                self.window_size,
                self.dft_length + self.norm_offset,
                self.norm_offset,
                self.inverse_sqrt_win_size,
            )
        return self._bags(mft)

    def fit_transform(self, X, y=None):
        """Fit and transform sharing the MFT computation."""
        self.fit(X, y)
        return self._bags(self._fit_mft)

    def binning_bags(self):
        """Bags built from the binning DFT (reduced bags for dim selection)."""
        if self._binning_dft is None:
            raise ValueError("fit with keep_binning_dft=True first")
        return self._bags(self._binning_dft)

    def _bags(self, dfts):
        return _bags_from_dft(
            dfts,
            self.breakpoints,
            self.word_length,
            self.word_bits,
            self.window_size,
            self.n_timepoints,
            self.levels,
            self.bigrams,
        )

    def _check_X(self, X):
        if X.ndim == 3:
            if X.shape[1] != 1:
                raise ValueError("only univariate input, slice channels in TDE")
            X = X.reshape(X.shape[0], X.shape[2])
        return np.ascontiguousarray(X, dtype=np.float64)

    def _mcb_equi_depth(self, dft):
        total = dft.shape[0]
        breakpoints = np.zeros((self.word_length, ALPHABET_SIZE))
        target_bin_depth = total / ALPHABET_SIZE

        for letter in range(self.word_length):
            # 2dp rounding retained from the original implementation
            column = np.sort(np.rint(dft[:, letter] * 100) / 100)
            bin_index = 0.0
            for bp in range(ALPHABET_SIZE - 1):
                bin_index += target_bin_depth
                breakpoints[letter, bp] = column[int(bin_index)]

        breakpoints[:, ALPHABET_SIZE - 1] = _DBL_MAX
        return breakpoints

    def _igb(self, dft, y):
        y = np.asarray(y)
        _, y_codes = np.unique(y, return_inverse=True)
        dft = dft[:, : self.word_length].astype(np.float32).astype(np.float64)
        thresholds, n_thresholds = _igb_all(
            dft,
            y_codes.astype(np.int64),
            int(y_codes.max()) + 1,
        )

        breakpoints = np.full((self.word_length, ALPHABET_SIZE), _DBL_MAX)
        for letter in range(self.word_length):
            for bp in range(n_thresholds[letter]):
                breakpoints[letter, bp] = thresholds[letter, bp]
        return np.sort(breakpoints, axis=1)


@njit(fastmath=True, cache=True)
def _incremental_stds(series, end, window_size):
    stds = np.zeros(end)
    series_sum = 0.0
    square_sum = 0.0
    for i in range(window_size):
        series_sum += series[i]
        square_sum += series[i] * series[i]

    r_window_length = 1.0 / window_size
    mean = series_sum * r_window_length
    buf = math.sqrt(square_sum * r_window_length - mean * mean)
    stds[0] = buf if buf > 1e-8 else 1.0

    for w in range(1, end):
        series_sum += series[w + window_size - 1] - series[w - 1]
        mean = series_sum * r_window_length
        square_sum += (
            series[w + window_size - 1] * series[w + window_size - 1]
            - series[w - 1] * series[w - 1]
        )
        buf = math.sqrt(square_sum * r_window_length - mean * mean)
        stds[w] = buf if buf > 1e-8 else 1.0

    return stds


@njit(fastmath=True, cache=True)
def _mft_all(X, window_size, length, norm_offset, inverse_sqrt_win_size):
    """Normalised sliding-window Fourier coefficients for every case.

    Returns a (n_cases, n_windows, length - norm_offset) array: the first
    norm_offset values (the first coefficient pair when norm is used) are
    dropped from the output.
    """
    n_cases, n_timepoints = X.shape
    end = max(1, n_timepoints - window_size + 1)
    half = length // 2

    phis = np.zeros(length)
    for i in range(half):
        phis[i * 2] = math.cos(2 * math.pi * (-i) / window_size)
        phis[i * 2 + 1] = -math.sin(2 * math.pi * (-i) / window_size)

    out_len = length - norm_offset
    out = np.zeros((n_cases, end, out_len))
    mft = np.zeros(length)

    for c in range(n_cases):
        series = X[c]
        stds = _incremental_stds(series, end, window_size)

        # first window: direct DFT, O(window_size * length)
        for i in range(half):
            real = 0.0
            imag = 0.0
            for t in range(window_size):
                angle = 2 * math.pi * t * i / window_size
                real += series[t] * math.cos(angle)
                imag += -series[t] * math.sin(angle)
            mft[i * 2] = real
            mft[i * 2 + 1] = imag

        factor = inverse_sqrt_win_size / stds[0]
        for j in range(out_len):
            out[c, 0, j] = mft[norm_offset + j] * factor

        # remaining windows: incremental MFT update
        for w in range(1, end):
            diff = series[w + window_size - 1] - series[w - 1]
            for i2 in range(0, length, 2):
                real = mft[i2] + diff
                imag = mft[i2 + 1]
                mft[i2] = real * phis[i2] - imag * phis[i2 + 1]
                mft[i2 + 1] = real * phis[i2 + 1] + phis[i2] * imag

            factor = inverse_sqrt_win_size / stds[w]
            for j in range(out_len):
                out[c, w, j] = mft[norm_offset + j] * factor

    return out


@njit(fastmath=True, cache=True)
def _binning_dft_all(
    X, window_size, dft_length, norm, inverse_sqrt_win_size, num_windows_per_inst
):
    n_cases, n_timepoints = X.shape
    start = 2 if norm else 0
    output_length = start + dft_length
    c = start // 2

    out = np.zeros((n_cases, num_windows_per_inst, dft_length))
    for case in range(n_cases):
        series = X[case]
        for window in range(num_windows_per_inst):
            if window == num_windows_per_inst - 1:
                offset = n_timepoints - window_size
            else:
                offset = window * window_size

            series_sum = 0.0
            square_sum = 0.0
            for n in range(window_size):
                value = series[offset + n]
                series_sum += value
                square_sum += value * value

            mean = series_sum / window_size
            std = math.sqrt(square_sum / window_size - mean * mean)
            if std == 0.0:
                std = 1.0
            factor = inverse_sqrt_win_size / std

            for i in range(c, output_length // 2):
                real = 0.0
                imag = 0.0
                for n in range(window_size):
                    value = series[offset + n]
                    angle = 2 * math.pi * n * i / window_size
                    real += value * math.cos(angle)
                    imag += -value * math.sin(angle)
                out[case, window, (i - c) * 2] = real * factor
                out[case, window, (i - c) * 2 + 1] = imag * factor

    return out


@njit(cache=True)
def _bags_from_dft(
    dfts,
    breakpoints,
    word_length,
    word_bits,
    window_size,
    n_timepoints,
    levels,
    bigrams,
):
    """Words and aggregated bags for all cases from their window DFTs.

    Numerosity reduction is always applied; the alphabet is fixed at 4
    (2 bits per letter). Output bags are sorted lexicographically by
    (key1, key2).

    Bag events carry no explicit values: a bigram always counts 1 and a
    pyramid unigram counts 2**level, which is recoverable from its quadrant.
    With levels <= 3 a unigram event packs into a single int64
    ((word << 3) | quadrant), so per-case bags reduce to plain np.sort of
    key arrays followed by run-length aggregation.

    - levels == 1: unigram words and packed bigrams share one key space
      (key2 = 0), exactly like the flat typed-dict SFA, so a bigram with
      previous word 0 merges with the unigram of the same value.
    - levels > 1: unigrams are packed (word << 3) | quadrant; bigrams are
      kept separately with key2 = -1 and merged in during aggregation.
    """
    n_cases, n_windows, _ = dfts.shape
    events_per_case = n_windows * (levels + (1 if bigrams else 0))

    keys1 = np.empty(n_cases * events_per_case, dtype=np.int64)
    keys2 = np.empty(n_cases * events_per_case, dtype=np.int64)
    counts = np.empty(n_cases * events_per_case, dtype=np.uint32)
    offsets = np.zeros(n_cases + 1, dtype=np.int64)

    words = np.zeros(n_windows, dtype=np.int64)
    # when levels == 1 bigrams share the unigram key space and array
    uni = np.empty(n_windows * (levels + (1 if bigrams else 0)), dtype=np.int64)
    big = np.empty(n_windows if bigrams else 0, dtype=np.int64)

    pos = 0
    for c in range(n_cases):
        # one word per window; letter = number of breakpoints below the
        # value (branchless, alphabet 4, last breakpoint is DBL_MAX)
        for wi in range(n_windows):
            word = np.int64(0)
            for i in range(word_length):
                v = dfts[c, wi, i]
                letter = (
                    np.int64(v > breakpoints[i, 0])
                    + np.int64(v > breakpoints[i, 1])
                    + np.int64(v > breakpoints[i, 2])
                )
                word = (word << LETTER_BITS) | letter
            words[wi] = word

        # emit key events, numerosity reduction always on
        n_u = 0
        n_b = 0
        last_word = np.int64(-1)
        repeat_words = 0
        for wi in range(n_windows):
            word = words[wi]

            if word == last_word:
                repeat_words += 1
            else:
                if levels > 1:
                    window_ind = wi - repeat_words // 2
                    start = 0
                    for level in range(levels):
                        num_quadrants = 2**level
                        quadrant = start + (window_ind + window_size // 2) // (
                            n_timepoints // num_quadrants
                        )
                        uni[n_u] = (word << 3) | quadrant
                        n_u += 1
                        start += num_quadrants
                else:
                    uni[n_u] = word
                    n_u += 1
                last_word = word
                repeat_words = 0

            if bigrams and wi - window_size >= 0:
                bigram = (words[wi - window_size] << word_bits) | word
                if levels > 1:
                    big[n_b] = bigram
                    n_b += 1
                else:
                    # shared key space with unigrams, matching the flat
                    # typed-dict SFA
                    uni[n_u] = bigram
                    n_u += 1

        su = np.sort(uni[:n_u])
        if levels > 1:
            sb = np.sort(big[:n_b])
        else:
            sb = big[:0]
            n_b = 0

        # merge-aggregate the two sorted streams; for a shared key1 the
        # bigram (key2 = -1) sorts before any unigram (key2 >= 0)
        i = 0
        j = 0
        while i < n_u or j < n_b:
            if j < n_b and (i >= n_u or sb[j] <= su[i] >> 3):
                # bigram run
                bk = sb[j]
                run = 1
                j += 1
                while j < n_b and sb[j] == bk:
                    run += 1
                    j += 1
                keys1[pos] = bk
                keys2[pos] = -1
                counts[pos] = run
                pos += 1
            else:
                # unigram run
                uk = su[i]
                run = 1
                i += 1
                while i < n_u and su[i] == uk:
                    run += 1
                    i += 1
                if levels > 1:
                    quadrant = uk & 7
                    # weight 2**level of the quadrant: 0 -> 1, 1-2 -> 2,
                    # 3-6 -> 4
                    if quadrant == 0:
                        weight = 1
                    elif quadrant <= 2:
                        weight = 2
                    else:
                        weight = 4
                    keys1[pos] = uk >> 3
                    keys2[pos] = quadrant
                    counts[pos] = run * weight
                else:
                    keys1[pos] = uk
                    keys2[pos] = 0
                    counts[pos] = run
                pos += 1

        offsets[c + 1] = pos

    return keys1[:pos].copy(), keys2[:pos].copy(), counts[:pos].copy(), offsets


@njit(cache=True)
def _entropy(class_counts, n):
    h = 0.0
    for k in range(len(class_counts)):
        if class_counts[k] > 0:
            p = class_counts[k] / n
            h -= p * math.log(p)
    return h


@njit(cache=True)
def _best_split(xs, ys, start, end, n_classes, n_total):
    """Best entropy split of sorted segment [start, end).

    Returns (improvement, split_pos, threshold); split_pos is the index of
    the last element of the left child, or -1 if no valid split exists.
    Improvement is sklearn's weighted impurity decrease (n_node / n_total) *
    (H - weighted child H), so it is comparable across nodes.
    """
    n_node = end - start
    if n_node < 2:
        return -1.0, -1, 0.0

    counts = np.zeros(n_classes)
    for i in range(start, end):
        counts[ys[i]] += 1
    h_node = _entropy(counts, n_node)
    if h_node <= 1e-12:
        return -1.0, -1, 0.0

    left = np.zeros(n_classes)
    right = counts.copy()
    best_gain = -1.0
    best_pos = -1
    best_thr = 0.0
    n_left = 0

    for i in range(start, end - 1):
        left[ys[i]] += 1
        right[ys[i]] -= 1
        n_left += 1
        if xs[i + 1] > xs[i]:
            n_right = n_node - n_left
            weighted = (
                n_left * _entropy(left, n_left) + n_right * _entropy(right, n_right)
            ) / n_node
            gain = (n_node / n_total) * (h_node - weighted)
            if gain > best_gain:
                best_gain = gain
                best_pos = i
                thr = (xs[i] + xs[i + 1]) / 2.0
                # guard against midpoint rounding up to the right value
                if thr == xs[i + 1]:
                    thr = xs[i]
                best_thr = thr

    return best_gain, best_pos, best_thr


@njit(cache=True)
def _igb_all(dft, y_codes, n_classes):
    """Information gain binning for every letter, alphabet fixed at 4.

    Best-first growth of an entropy decision tree on one feature until 4
    leaves (or no further valid splits), depth-limited to 2, the same
    procedure sklearn's DecisionTreeClassifier uses with max_leaf_nodes=4
    and max_depth=2.
    """
    max_depth = 2
    n_letters = dft.shape[1]
    n_total = dft.shape[0]

    thresholds = np.zeros((n_letters, ALPHABET_SIZE - 1))
    n_thresholds = np.zeros(n_letters, dtype=np.int64)

    max_cand = 2 * ALPHABET_SIZE
    c_start = np.zeros(max_cand, dtype=np.int64)
    c_end = np.zeros(max_cand, dtype=np.int64)
    c_depth = np.zeros(max_cand, dtype=np.int64)
    c_pos = np.zeros(max_cand, dtype=np.int64)
    c_thr = np.zeros(max_cand)
    c_gain = np.zeros(max_cand)
    c_active = np.zeros(max_cand, dtype=np.bool_)

    for letter in range(n_letters):
        col = dft[:, letter]
        order = np.argsort(col)
        xs = col[order]
        ys = y_codes[order]

        n_cand = 0
        c_active[:] = False

        gain, pos, thr = _best_split(xs, ys, 0, n_total, n_classes, n_total)
        if pos >= 0:
            c_start[0] = 0
            c_end[0] = n_total
            c_depth[0] = 0
            c_pos[0] = pos
            c_thr[0] = thr
            c_gain[0] = gain
            c_active[0] = True
            n_cand = 1

        n_leaves = 1
        n_thr = 0
        while n_leaves < ALPHABET_SIZE:
            # pick the active candidate with the highest improvement
            best = -1
            best_gain = -1.0
            for k in range(n_cand):
                if c_active[k] and c_gain[k] > best_gain:
                    best_gain = c_gain[k]
                    best = k
            if best < 0:
                break

            c_active[best] = False
            thresholds[letter, n_thr] = c_thr[best]
            n_thr += 1
            n_leaves += 1

            # evaluate the two children as new candidates
            child_depth = c_depth[best] + 1
            if child_depth < max_depth:
                s = c_start[best]
                e = c_end[best]
                mid = c_pos[best] + 1
                for lo, hi in ((s, mid), (mid, e)):
                    gain, pos, thr = _best_split(xs, ys, lo, hi, n_classes, n_total)
                    if pos >= 0 and n_cand < max_cand:
                        c_start[n_cand] = lo
                        c_end[n_cand] = hi
                        c_depth[n_cand] = child_depth
                        c_pos[n_cand] = pos
                        c_thr[n_cand] = thr
                        c_gain[n_cand] = gain
                        c_active[n_cand] = True
                        n_cand += 1

        n_thresholds[letter] = n_thr

    return thresholds, n_thresholds


@njit(cache=True)
def _histogram_intersection(keys1, keys2, counts, a0, a1, b0, b1):
    """Merge intersection of two sorted bag segments (sum of min counts)."""
    sim = 0
    i, j = a0, b0
    while i < a1 and j < b1:
        ka1, ka2 = keys1[i], keys2[i]
        kb1, kb2 = keys1[j], keys2[j]
        if ka1 == kb1 and ka2 == kb2:
            sim += min(counts[i], counts[j])
            i += 1
            j += 1
        elif ka1 < kb1 or (ka1 == kb1 and ka2 < kb2):
            i += 1
        else:
            j += 1
    return sim


@njit(cache=True)
def nn_predict_loocv(keys1, keys2, counts, offsets, train_num):
    """Index of the 1NN of bag train_num among all other bags."""
    a0, a1 = offsets[train_num], offsets[train_num + 1]
    best_sim = -1
    nn = -1
    for n in range(len(offsets) - 1):
        if n == train_num:
            continue
        sim = _histogram_intersection(
            keys1, keys2, counts, a0, a1, offsets[n], offsets[n + 1]
        )
        if sim > best_sim:
            best_sim = sim
            nn = n
    return nn


@njit(cache=True)
def nn_similarities(keys1, keys2, counts, offsets, t_keys1, t_keys2, t_counts, t0, t1):
    """Similarities of one test bag segment against all train bags."""
    n = len(offsets) - 1
    sims = np.zeros(n, dtype=np.int64)
    for m in range(n):
        b0, b1 = offsets[m], offsets[m + 1]
        sim = 0
        i, j = t0, b0
        while i < t1 and j < b1:
            ka1, ka2 = t_keys1[i], t_keys2[i]
            kb1, kb2 = keys1[j], keys2[j]
            if ka1 == kb1 and ka2 == kb2:
                sim += min(t_counts[i], counts[j])
                i += 1
                j += 1
            elif ka1 < kb1 or (ka1 == kb1 and ka2 < kb2):
                i += 1
            else:
                j += 1
        sims[m] = sim
    return sims
