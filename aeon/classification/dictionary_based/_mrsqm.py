"""Multiple Representations Sequence Miner (MrSQM) Classifier."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["MrSQMClassifier"]

import math

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from numba.typed import List
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from aeon.classification import BaseClassifier
from aeon.utils.validation import check_n_jobs


class MrSQMClassifier(BaseClassifier):
    """
    Multiple Representations Sequence Miner (MrSQM) classifier.

    MrSQM transforms each time series into multiple symbolic representations using
    sliding window SAX and SFA with randomly selected window sizes, word lengths and
    alphabet sizes. Subsequences of the symbolic words are selected as features using
    one of four strategies, and a logistic regression classifier is trained on the
    binary presence of each selected subsequence.

    The feature selection strategies are:

    - ``"R"``: randomly sample ``features_per_rep`` subsequences from the words of
      each representation.
    - ``"S"``: mine the ``features_per_rep`` subsequences with the highest chi-squared
      score using a branch-and-bound search (SQM).
    - ``"RS"``: randomly sample ``selection_per_rep`` subsequences, then keep the
      ``features_per_rep`` with the highest chi-squared score.
    - ``"SR"``: mine ``selection_per_rep`` subsequences with SQM, then randomly keep
      ``features_per_rep`` of them.

    This is a native implementation of the ``mrsqm`` package by the original authors
    [1]_. For the same ``random_state`` it follows the same sequence of random draws
    and the same symbolic transformations as ``mrsqm`` 0.0.7.

    Parameters
    ----------
    strat : str, default="RS"
        Feature selection strategy. One of "R", "S", "RS" or "SR". "R" and "S" are
        single-stage filters while "RS" and "SR" are two-stage filters.
    features_per_rep : int, default=500
        The (maximum) number of features selected per representation.
    selection_per_rep : int, default=2000
        The (maximum) number of candidate features selected per representation.
        Only applied in two stages strategies ("RS" and "SR").
    nsax : int, default=0
        Controls the number of representations produced by the SAX transformation.
        ``nsax * floor(log2(n_timepoints))`` representations are sampled for each
        channel.
    nsfa : int, default=5
        Controls the number of representations produced by the SFA transformation.
        ``nsfa * floor(log2(n_timepoints))`` representations are sampled for each
        channel.
    sfa_norm : bool, default=True
        Whether to z-normalise each series before the SFA transformation.
    first_diff : bool, default=True
        If True, each SFA representation randomly uses either the series or its first
        order differences.
    custom_config : list of dict, default=None
        Custom configuration for the symbolic transformations, replacing the randomly
        sampled one. Each dict must contain a "method" ("sax" or "sfa"), "window",
        "word" and "alphabet" key. SAX configurations may contain "dilation" (default
        1). SFA configurations may contain "normSFA" (default False), "normTS"
        (default ``sfa_norm``) and "diff" (default False).
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, a seed for the random number generator is drawn
        from it;
        If `None`, the random number generator is seeded randomly.
    n_jobs : int, default=1
        The number of jobs to run in parallel for the symbolic transformations and
        feature extraction. ``-1`` means using all processors.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The unique class labels.

    See Also
    --------
    WEASEL_V2, MUSE

    Notes
    -----
    The original implementation is available at https://github.com/mlgig/mrsqm.

    The original computes the discrete Fourier transform for SFA with FFTW, while
    this implementation uses numpy. When the Fourier coefficients of a window are
    zero up to floating point error, such as in flat regions of a series, the SFA
    symbol can depend on the rounding of the FFT library, and may differ from the
    original. Otherwise, the symbolic words and selected features are identical to
    the original for the same ``random_state``.

    References
    ----------
    .. [1] Nguyen, Thach Le, and Georgiana Ifrim. "Fast time series classification with
        random symbolic subsequences." Advanced Analytics and Learning on Temporal Data:
        7th ECML PKDD Workshop, AALTD 2022, Grenoble, France, September 19–23, 2022.

    Examples
    --------
    >>> from aeon.classification.dictionary_based import MrSQMClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(n_cases=10, n_timepoints=32, random_state=0)
    >>> clf = MrSQMClassifier(random_state=0)
    >>> clf.fit(X, y)
    MrSQMClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        strat: str = "RS",
        features_per_rep: int = 500,
        selection_per_rep: int = 2000,
        nsax: int = 0,
        nsfa: int = 5,
        sfa_norm: bool = True,
        first_diff: bool = True,
        custom_config: list | None = None,
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.strat = strat
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        self.nsax = nsax
        self.nsfa = nsfa
        self.sfa_norm = sfa_norm
        self.first_diff = first_diff
        self.custom_config = custom_config
        self.random_state = random_state
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X, y):
        if self.strat not in ("R", "S", "RS", "SR"):
            raise ValueError(
                f"strat must be one of 'R', 'S', 'RS' or 'SR', found {self.strat}."
            )

        self._n_jobs = check_n_jobs(self.n_jobs)
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        X = np.asarray(X, dtype=np.float64)
        rng = self._get_rng()
        y_idx = np.array([self._class_dictionary[c] for c in y], dtype=np.int64)

        if self.custom_config is None:
            self._config = self._generate_config(X.shape[2], rng)
        else:
            self._config = self._validate_custom_config(X.shape[2])
        if len(self._config) == 0:
            raise ValueError(
                "No symbolic representations were created. Series must have at "
                "least 8 time points, and at least one of nsax or nsfa must be "
                "positive."
            )

        X_diff = np.diff(X, axis=2, prepend=0) if self._uses_diff() else None
        self._features = []
        blocks = []
        for cfg in self._config:
            for c in range(X.shape[1]):
                words, n_words = self._symbolic_words(X, X_diff, cfg, c, fit=True)
                features = self._mine(words, n_words, y_idx, rng)
                fm = _feature_presence(words, n_words, *_features_to_array(features))
                if self.strat == "RS":
                    fs = SelectKBest(chi2, k=min(self.features_per_rep, fm.shape[1]))
                    fm = fs.fit_transform(fm, y_idx)
                    features = [features[i] for i in fs.get_support(indices=True)]
                self._features.append(features)
                blocks.append(fm)

        # sklearn >= 1.8 removed multi_class="multinomial", which the original used.
        # Multinomial is the default for more than two classes. For two classes, the
        # multinomial model with C=1 has the same optimum as binary logistic
        # regression with C=2.
        self.clf_ = LogisticRegression(
            solver="newton-cg",
            C=2.0 if self.n_classes_ == 2 else 1.0,
            class_weight="balanced",
            random_state=0,
        )
        self.clf_.fit(np.hstack(blocks), y)

        set_num_threads(prev_threads)
        return self

    def _predict(self, X) -> np.ndarray:
        return self.clf_.predict(self._transform_features(X))

    def _predict_proba(self, X) -> np.ndarray:
        return self.clf_.predict_proba(self._transform_features(X))

    def _transform_features(self, X):
        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        X = np.asarray(X, dtype=np.float64)
        X_diff = np.diff(X, axis=2, prepend=0) if self._uses_diff() else None
        blocks = []
        i = 0
        for cfg in self._config:
            for c in range(X.shape[1]):
                words, n_words = self._symbolic_words(X, X_diff, cfg, c)
                blocks.append(
                    _feature_presence(
                        words, n_words, *_features_to_array(self._features[i])
                    )
                )
                i += 1

        set_num_threads(prev_threads)
        return np.hstack(blocks)

    def _get_rng(self):
        # the original uses np.random.default_rng(random_state), an int seed gives the
        # same stream as the original
        rs = self.random_state
        if rs is None or isinstance(rs, (int, np.integer)):
            return np.random.default_rng(rs)
        return np.random.default_rng(
            check_random_state(rs).randint(np.iinfo(np.int32).max)
        )

    def _generate_config(self, n_timepoints, rng):
        # order of random draws matches MrSQMTransformer.transform_time_series
        config = []
        for p in _create_pars(n_timepoints, self.nsax, rng):
            config.append(
                {
                    "method": "sax",
                    "window": p[0],
                    "word": p[1],
                    "alphabet": p[2],
                    "dilation": 1,
                }
            )
        for p in _create_pars(n_timepoints, self.nsfa, rng):
            config.append(
                {
                    "method": "sfa",
                    "window": p[0],
                    "word": p[1],
                    "alphabet": p[2],
                    "normSFA": False,
                    "normTS": self.sfa_norm,
                    "diff": bool(rng.choice([True, False])),
                    "signature": [],
                }
            )
        return config

    def _validate_custom_config(self, n_timepoints):
        config = []
        for cfg in self.custom_config:
            cfg = dict(cfg)
            if cfg["method"] == "sax":
                cfg.setdefault("dilation", 1)
                if cfg["alphabet"] not in _SAX_BREAKPOINTS:
                    raise ValueError("SAX alphabet size must be between 2 and 16.")
                span = (cfg["window"] - 1) * cfg["dilation"] + 1
            elif cfg["method"] == "sfa":
                cfg.setdefault("normSFA", False)
                cfg.setdefault("normTS", self.sfa_norm)
                cfg.setdefault("diff", False)
                cfg["signature"] = []
                span = cfg["window"]
                if cfg["word"] + 2 * cfg["normSFA"] > cfg["window"]:
                    raise ValueError("SFA window must be larger than the word length.")
            else:
                raise ValueError(f"Unknown symbolic method {cfg['method']}.")
            if span > n_timepoints:
                raise ValueError(
                    "Custom configuration windows must not be longer than the series."
                )
            config.append(cfg)
        return config

    def _uses_diff(self):
        return self.first_diff and any(
            cfg["method"] == "sfa" and cfg["diff"] for cfg in self._config
        )

    def _symbolic_words(self, X, X_diff, cfg, channel, fit=False):
        if cfg["method"] == "sax":
            return _sax_words(
                X[:, channel, :],
                cfg["window"],
                cfg["word"],
                _SAX_BREAKPOINTS[cfg["alphabet"]],
                cfg["dilation"],
            )

        X_c = (
            X_diff[:, channel, :]
            if cfg["diff"] and self.first_diff
            else X[:, channel, :]
        )
        # the original fits one lookup table per channel and appends one to the
        # config for each channel. The SFA word has one extra coefficient which is
        # skipped when creating the word.
        n_coefs = cfg["word"] + 1
        start_offset = 2 if cfg["normSFA"] else 0
        X_c = _normalise_rows(X_c, cfg["normTS"])
        if fit:
            cfg["signature"].append(
                _sfa_lookup_table(
                    X_c,
                    cfg["window"],
                    n_coefs,
                    cfg["alphabet"],
                    cfg["normSFA"],
                    start_offset,
                )
            )
        first_dft = np.fft.rfft(X_c[:, : cfg["window"]], axis=1)
        return _sfa_words(
            X_c,
            cfg["window"],
            n_coefs,
            cfg["alphabet"],
            start_offset,
            cfg["signature"][channel],
            np.ascontiguousarray(first_dft.real),
            np.ascontiguousarray(first_dft.imag),
        )

    def _mine(self, words, n_words, y_idx, rng):
        if self.strat == "S":
            return _sqm_mine(words, n_words, y_idx, self.features_per_rep)
        elif self.strat == "SR":
            mined = _sqm_mine(words, n_words, y_idx, self.selection_per_rep)
            return rng.permutation(mined)[: self.features_per_rep].tolist()
        elif self.strat == "R":
            return _sample_random_sequences(
                words, n_words, 3, 16, self.features_per_rep, rng
            )
        else:
            return _sample_random_sequences(
                words, n_words, 3, 16, self.selection_per_rep, rng
            )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict | list[dict]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "features_per_rep": 50,
            "selection_per_rep": 200,
            "nsax": 1,
            "nsfa": 1,
        }


# SAX breakpoints for alphabet sizes 2 to 16, as used by the original.
_SAX_BREAKPOINTS = {
    2: np.array([0.0]),
    3: np.array([-0.430727299295, 0.430727299295]),
    4: np.array([-0.674489750196, 0.0, 0.674489750196]),
    5: np.array([-0.841621233573, -0.253347103136, 0.253347103136, 0.841621233573]),
    6: np.array(
        [-0.967421566102, -0.430727299295, 0.0, 0.430727299295, 0.967421566102]
    ),
    7: np.array(
        [
            -1.06757052388,
            -0.565948821933,
            -0.180012369793,
            0.180012369793,
            0.565948821933,
            1.06757052388,
        ]
    ),
    8: np.array(
        [
            -1.15034938038,
            -0.674489750196,
            -0.318639363964,
            0.0,
            0.318639363964,
            0.674489750196,
            1.15034938038,
        ]
    ),
    9: np.array(
        [
            -1.22064034885,
            -0.764709673786,
            -0.430727299295,
            -0.139710298882,
            0.139710298882,
            0.430727299295,
            0.764709673786,
            1.22064034885,
        ]
    ),
    10: np.array(
        [
            -1.28155156554,
            -0.841621233573,
            -0.524400512708,
            -0.253347103136,
            0.0,
            0.253347103136,
            0.524400512708,
            0.841621233573,
            1.28155156554,
        ]
    ),
    11: np.array(
        [
            -1.33517773612,
            -0.908457868537,
            -0.604585346583,
            -0.348755695517,
            -0.114185294321,
            0.114185294321,
            0.348755695517,
            0.604585346583,
            0.908457868537,
            1.33517773612,
        ]
    ),
    12: np.array(
        [
            -1.3829941271,
            -0.967421566102,
            -0.674489750196,
            -0.430727299295,
            -0.210428394248,
            0.0,
            0.210428394248,
            0.430727299295,
            0.674489750196,
            0.967421566102,
            1.3829941271,
        ]
    ),
    13: np.array(
        [
            -1.42607687227,
            -1.02007623279,
            -0.736315917376,
            -0.502402223373,
            -0.293381232121,
            -0.0965586152896,
            0.0965586152896,
            0.293381232121,
            0.502402223373,
            0.736315917376,
            1.02007623279,
            1.42607687227,
        ]
    ),
    14: np.array(
        [
            -1.46523379269,
            -1.06757052388,
            -0.791638607743,
            -0.565948821933,
            -0.366106356801,
            -0.180012369793,
            0.0,
            0.180012369793,
            0.366106356801,
            0.565948821933,
            0.791638607743,
            1.06757052388,
            1.46523379269,
        ]
    ),
    15: np.array(
        [
            -1.50108594604,
            -1.11077161664,
            -0.841621233573,
            -0.62292572321,
            -0.430727299295,
            -0.253347103136,
            -0.0836517339071,
            0.0836517339071,
            0.253347103136,
            0.430727299295,
            0.62292572321,
            0.841621233573,
            1.11077161664,
            1.50108594604,
        ]
    ),
    16: np.array(
        [
            -1.53412054435,
            -1.15034938038,
            -0.887146559019,
            -0.674489750196,
            -0.488776411115,
            -0.318639363964,
            -0.15731068461,
            0.0,
            0.15731068461,
            0.318639363964,
            0.488776411115,
            0.674489750196,
            0.887146559019,
            1.15034938038,
            1.53412054435,
        ]
    ),
}

# the original uses this truncated value of pi for the MFT
_MFT_PI = 3.14159265
_SPACE = 32


def _create_pars(max_ws, xrep, rng):
    """Sample (window, word length, alphabet size) triples for xrep."""
    pars = []
    if xrep > 0:
        candidates = []
        for wd in [
            int(2 ** (w / xrep))
            for w in range(3 * xrep, xrep * int(np.log2(max_ws)) + 1)
        ]:
            for wo in [7, 8]:
                for alphabet in [2, 3, 4, 5]:
                    if wo <= (wd + 1):
                        candidates.append([wd, wo, alphabet])

        nrep = xrep * int(np.log2(max_ws))
        if nrep >= len(candidates):
            pars = candidates
        else:
            selected = rng.choice(range(len(candidates)), replace=False, size=nrep)
            pars = [candidates[i] for i in selected]
    return pars


def _sample_random_sequences(words, n_words, min_length, max_length, max_n_seq, rng):
    """Randomly sample unique subsequences of symbolic words."""
    # the draws must match the original exactly to select the same features
    output = {}
    n_input = words.shape[0]
    word_length = words.shape[2]
    for _ in range(max_n_seq):
        did = rng.integers(0, high=n_input)
        wid = rng.integers(0, high=int(n_words[did]))
        s_length = rng.integers(min_length, high=min(word_length + 1, max_length + 1))
        start = rng.integers(0, high=word_length - s_length + 1)
        output[words[did, wid, start : start + s_length].tobytes()] = 1
    return list(output.keys())


def _features_to_array(features):
    lengths = np.array([len(f) for f in features], dtype=np.int64)
    chars = np.zeros(
        (len(features), lengths.max() if len(features) > 0 else 0), dtype=np.uint8
    )
    for i, f in enumerate(features):
        chars[i, : lengths[i]] = np.frombuffer(f, dtype=np.uint8)
    return chars, lengths


@njit(cache=True, fastmath=False)
def _c_round(x):
    # C round, halfway cases away from zero
    r = np.trunc(x)
    if abs(x - r) >= 0.5:
        r += math.copysign(1.0, x)
    return r


@njit(cache=True, fastmath=False)
def _norm_series(x, norm_mean):
    # TimeSeries::norm, uses one-pass variance
    n = x.shape[0]
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n
    var = 0.0
    for i in range(n):
        var += x[i] * x[i]
    buf = (1.0 / n) * var - mean * mean
    std = math.sqrt(buf) if buf > 0 else 0.0
    inv_std = 1.0 / (std if std > 0 else 1.0)
    if norm_mean:
        for i in range(n):
            x[i] = (x[i] - mean) * inv_std
    elif inv_std != 1.0:
        for i in range(n):
            x[i] *= inv_std


@njit(cache=True, fastmath=False, parallel=True)
def _normalise_rows(X, normalise):
    X = X.copy()
    if normalise:
        for i in prange(X.shape[0]):
            _norm_series(X[i], True)
    return X


@njit(cache=True, fastmath=False, parallel=True)
def _sax_words(X, window, word_length, breakpoints, dilation):
    """Sliding window SAX with back-to-back numerosity reduction."""
    n_cases, n_timepoints = X.shape
    n_windows = max(n_timepoints - (window - 1) * dilation, 0)
    words = np.empty((n_cases, max(n_windows, 1), word_length), dtype=np.uint8)
    n_words = np.zeros(n_cases, dtype=np.int64)
    paa_size = window * 1.0 / word_length

    for i in prange(n_cases):
        subsection = np.empty(window)
        word = np.empty(word_length, dtype=np.uint8)
        for cur in range(n_windows):
            s = 0.0
            sq = 0.0
            for k in range(window):
                v = X[i, cur + k * dilation]
                s += v
                sq += v * v
            mean = s / window
            var = sq / window - mean * mean
            for k in range(window):
                v = X[i, cur + k * dilation] - mean
                if var > 0:
                    v = v / math.sqrt(var)
                subsection[k] = v

            for w in range(word_length):
                start_idx = paa_size * w
                end_idx = paa_size * (w + 1) - 1
                full_start = int(math.ceil(start_idx))
                full_end = int(math.floor(end_idx))
                start_frac = full_start - start_idx
                end_frac = end_idx - full_end
                agg = 0.0
                if start_frac > 0:
                    agg += subsection[full_start - 1] * start_frac
                if end_frac > 0 and full_end < window - 1:
                    agg += subsection[full_end + 1] * end_frac
                for j in range(full_start, full_end + 1):
                    agg += subsection[j]
                paa = agg / paa_size

                b = 0
                for j in range(breakpoints.shape[0]):
                    if paa >= breakpoints[j]:
                        b += 1
                word[w] = 97 + b

            if n_words[i] > 0:
                same = True
                for w in range(word_length):
                    if words[i, n_words[i] - 1, w] != word[w]:
                        same = False
                        break
                if same:
                    continue
            words[i, n_words[i]] = word
            n_words[i] += 1

    return words, n_words


@njit(cache=True, fastmath=False, parallel=True)
def _sfa_disjoint_windows(X, window, norm_mean):
    n_cases, n_timepoints = X.shape
    amount = n_timepoints // window
    windows = np.empty((n_cases * amount, window))
    for i in prange(n_cases):
        for a in range(amount):
            w = X[i, a * window : (a + 1) * window].copy()
            _norm_series(w, norm_mean)
            windows[i * amount + a] = w
    return windows


@njit(cache=True, fastmath=False)
def _sfa_histogram_values(dft_re, dft_im, window, n_coefs, start_offset):
    n = dft_re.shape[0]
    norm = 1.0 / math.sqrt(window)
    offset = start_offset // 2
    m = min(n_coefs, window * 2 - start_offset)
    values = np.zeros((n, n_coefs + 1))
    for i in range(n):
        for k in range(0, m, 2):
            idx = k // 2 + offset
            values[i, k] = dft_re[i, idx] * norm
            values[i, k + 1] = -1 * (dft_im[i, idx] if idx > 0 else 0.0) * norm
        for k in range(n_coefs):
            values[i, k] = _c_round(values[i, k] * 100.0) / 100.0
    return values[:, :n_coefs]


def _sfa_lookup_table(X, window, n_coefs, alphabet, norm_mean, start_offset):
    """Equi-depth SFA breakpoints from a histogram of rounded coefficients."""
    windows = _sfa_disjoint_windows(X, window, norm_mean)
    table = np.full((n_coefs, alphabet - 1), np.inf)
    count = windows.shape[0]
    if count == 0:
        return table

    dft = np.fft.rfft(windows, axis=1)
    values = _sfa_histogram_values(
        np.ascontiguousarray(dft.real),
        np.ascontiguousarray(dft.imag),
        window,
        n_coefs,
        start_offset,
    )

    interval_size = count / float(alphabet)
    for j in range(n_coefs):
        # np.unique treats -0.0 and 0.0 as one value, like std::map<double, int>
        unique, counts = np.unique(values[:, j], return_counts=True)
        depth = 0
        beta = 0
        for v, c in zip(unique, counts):
            depth += c
            if depth > math.ceil(interval_size * (beta + 1)) and beta < alphabet - 1:
                table[j, beta] = v
                beta += 1
    return table


@njit(cache=True, fastmath=False, parallel=True)
def _sfa_words(X, window, n_coefs, alphabet, start_offset, table, first_re, first_im):
    """SFA words for each sliding window, using the MFT."""
    n_cases, n_timepoints = X.shape
    end = max(1, n_timepoints - window + 1)
    word_length = n_coefs - 1
    words = np.empty((n_cases, end, word_length), dtype=np.uint8)
    n_words = np.full(n_cases, end, dtype=np.int64)

    size = n_coefs + start_offset
    phis = np.zeros(size + 2)
    for u in range(0, size, 2):
        u_halve = float(-(u // 2))
        phis[u] = math.cos(2 * _MFT_PI * u_halve / window)
        phis[u + 1] = -math.sin(2 * _MFT_PI * u_halve / window)
    norm = 1.0 / math.sqrt(window)
    array_size = max(size, window)
    max_dft = first_re.shape[1]

    for i in prange(n_cases):
        x = X[i]

        # incremental inverse standard deviations of each window
        wl = min(n_timepoints, window)
        inv_stds = np.empty(n_timepoints - wl + 1)
        r = 1.0 / wl
        s = 0.0
        sq = 0.0
        for j in range(wl):
            s += x[j]
            sq += x[j] * x[j]
        mean = s * r
        buf = sq * r - mean * mean
        inv_stds[0] = 1.0 / math.sqrt(buf) if buf > 0 else 1.0
        for j in range(1, n_timepoints - wl + 1):
            s += x[j + wl - 1] - x[j - 1]
            mean = s * r
            sq += x[j + wl - 1] * x[j + wl - 1] - x[j - 1] * x[j - 1]
            buf = sq * r - mean * mean
            inv_stds[j] = 1.0 / math.sqrt(buf) if buf > 0 else 1.0

        mft = np.zeros(array_size + 2)
        mft2 = np.zeros(array_size + 2)
        for t in range(end):
            if t > 0:
                for k in range(start_offset, size, 2):
                    real1 = mft[k] + x[t + window - 1] - x[t - 1]
                    imag1 = mft[k + 1]
                    real = real1 * phis[k] - imag1 * phis[k + 1]
                    imag = real1 * phis[k + 1] + phis[k] * imag1
                    mft[k] = real
                    mft[k + 1] = imag
                    mft2[k - start_offset] = real
                    mft2[k - start_offset + 1] = imag
            else:
                for k in range(0, min(array_size, window * 2), 2):
                    idx = k // 2
                    if idx < max_dft:
                        mft[k] = first_re[i, idx]
                        mft[k + 1] = first_im[i, idx] if idx > 0 else 0.0
                for k in range(n_coefs):
                    mft2[k] = mft[k + start_offset]

            inv_std = norm * inv_stds[t]
            for k in range(0, n_coefs, 2):
                mft2[k] *= inv_std
                mft2[k + 1] *= -inv_std

            ai = 0
            for a in range(n_coefs):
                beta = 0
                while beta < alphabet - 1 and not mft2[a] < table[a, beta]:
                    beta += 1
                # the second value (imaginary part of the first coefficient) is
                # skipped
                if a != 1:
                    words[i, t, ai] = (33 + ai * alphabet + beta) % 256
                    ai += 1

    return words, n_words


@njit(cache=True, fastmath=False, parallel=True)
def _feature_presence(words, n_words, feature_chars, feature_lengths):
    """Binary presence of each feature as a substring of the symbolic words."""
    n_cases, _, word_length = words.shape
    n_features = feature_chars.shape[0]
    out = np.zeros((n_cases, n_features), dtype=np.bool_)
    if n_features == 0:
        return out

    char_map = np.full(256, -1, dtype=np.int64)
    n_chars = 0
    for f in range(n_features):
        for j in range(feature_lengths[f]):
            if char_map[feature_chars[f, j]] < 0:
                char_map[feature_chars[f, j]] = n_chars
                n_chars += 1

    # build a trie of the features
    children = np.full((feature_lengths.sum() + 1, n_chars), -1, dtype=np.int64)
    feature_at = np.full(feature_lengths.sum() + 1, -1, dtype=np.int64)
    n_nodes = 1
    for f in range(n_features):
        node = 0
        for j in range(feature_lengths[f]):
            c = char_map[feature_chars[f, j]]
            if children[node, c] < 0:
                children[node, c] = n_nodes
                n_nodes += 1
            node = children[node, c]
        feature_at[node] = f

    for i in prange(n_cases):
        if word_length <= 8:
            # consecutive windows often produce the same word, only search each
            # unique word of the case once
            keys = np.empty(n_words[i], dtype=np.uint64)
            for w in range(n_words[i]):
                key = np.uint64(0)
                for j in range(word_length):
                    key = (key << np.uint64(8)) | np.uint64(words[i, w, j])
                keys[w] = key
            order = np.argsort(keys)
            for k in range(n_words[i]):
                if k == 0 or keys[order[k]] != keys[order[k - 1]]:
                    _search_word(
                        words[i, order[k]], char_map, children, feature_at, out[i]
                    )
        else:
            for w in range(n_words[i]):
                _search_word(words[i, w], char_map, children, feature_at, out[i])
    return out


@njit(cache=True, fastmath=False)
def _search_word(word, char_map, children, feature_at, out):
    for s in range(word.shape[0]):
        node = 0
        for j in range(s, word.shape[0]):
            c = char_map[word[j]]
            if c < 0:
                break
            node = children[node, c]
            if node < 0:
                break
            if feature_at[node] >= 0:
                out[feature_at[node]] = True


def _sqm_mine(words, n_words, y_idx, n_features):
    """Mine the top chi-squared subsequences with SQM."""
    # class order and priors as in SQMiner::LabelManager, first appearance order
    _, first, inverse = np.unique(y_idx, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0])
    labels = rank[inverse].astype(np.int64)
    counts = np.bincount(labels)
    y_prob = counts * 1.0 / labels.shape[0]

    chars, lengths = _sqm_search(words, n_words, labels, y_prob, int(n_features))
    return [chars[i, : lengths[i]].tobytes() for i in range(chars.shape[0])]


@njit(cache=True, fastmath=False)
def _chi_square(observed, y_prob):
    feature_count = 0
    for o in observed:
        feature_count += o
    score = 0.0
    for i in range(observed.shape[0]):
        expected = y_prob[i] * feature_count
        d = observed[i] - expected
        score += d * d / expected
    return score


@njit(cache=True, fastmath=False)
def _chi_square_bound(observed, y_prob):
    bound = 0.0
    single = np.zeros_like(observed)
    for i in range(observed.shape[0]):
        if observed[i] > 0:
            single[:] = 0
            single[i] = observed[i]
            score = _chi_square(single, y_prob)
            if score >= bound:
                bound = score
    return bound


@njit(cache=True, fastmath=False)
def _char_at(words, doc, pos, word_length):
    k = pos // (word_length + 1)
    j = pos % (word_length + 1)
    if j == word_length:
        return _SPACE
    return words[doc, k, j]


@njit(cache=True, fastmath=False)
def _footprint_match(child, node):
    # same walk as SQMiner::expand_node, compares the documents of child and node
    ci = 0
    pi = 0
    nc = child.shape[0]
    nn = node.shape[0]
    while ci < nc and pi < nn and child[ci] == node[pi]:
        ci += 1
        pi += 1
        while ci < nc and child[ci] > 0 and ci < nc - 1:
            ci += 1
        while pi < nn and node[pi] > 0 and pi < nn - 1:
            pi += 1
        if pi == nn - 1 and ci == nc - 1:
            return True
    return False


@njit(cache=True, fastmath=False)
def _grow(a, size):
    b = np.empty(size, dtype=a.dtype)
    b[: a.shape[0]] = a
    return b


@njit(cache=True, fastmath=False)
def _sqm_search(words, n_words, labels, y_prob, capacity):
    """Port of SQMiner::mine with a fixed size store of top chi-squared nodes."""
    n_docs, _, word_length = words.shape
    n_classes = y_prob.shape[0]
    seq_len = n_words * (word_length + 1) - 1

    node_cap = 1024
    parent = np.empty(node_cap, dtype=np.int64)
    last_char = np.empty(node_cap, dtype=np.int64)
    chi = np.empty(node_cap)
    selected = np.zeros(node_cap, dtype=np.bool_)
    covered = np.zeros(node_cap, dtype=np.bool_)
    n_nodes = 0

    store = np.empty(capacity + 1, dtype=np.int64)
    store_n = 0

    stack_ids = List()
    stack_locs = List()

    # inverted index of unigrams, children of the root in character order
    pos_count = np.zeros(256, dtype=np.int64)
    doc_count = np.zeros(256, dtype=np.int64)
    last_doc = np.full(256, -1, dtype=np.int64)
    for d in range(n_docs):
        for p in range(seq_len[d]):
            c = _char_at(words, d, p, word_length)
            if c != _SPACE:
                pos_count[c] += 1
                if last_doc[c] != d:
                    doc_count[c] += 1
                    last_doc[c] = d
    locs = List()
    for c in range(256):
        locs.append(np.empty(pos_count[c] + doc_count[c], dtype=np.int64))
    fill = np.zeros(256, dtype=np.int64)
    last_doc[:] = -1
    for d in range(n_docs):
        for p in range(seq_len[d]):
            c = _char_at(words, d, p, word_length)
            if c != _SPACE:
                if last_doc[c] != d:
                    locs[c][fill[c]] = -(d + 1)
                    fill[c] += 1
                    last_doc[c] = d
                locs[c][fill[c]] = p
                fill[c] += 1
    for c in range(256):
        if pos_count[c] > 0:
            if n_nodes == node_cap:
                node_cap *= 2
                parent = _grow(parent, node_cap)
                last_char = _grow(last_char, node_cap)
                chi = _grow(chi, node_cap)
                selected = _grow(selected, node_cap)
                covered = _grow(covered, node_cap)
            parent[n_nodes] = -1
            last_char[n_nodes] = c
            selected[n_nodes] = False
            covered[n_nodes] = False
            stack_ids.append(n_nodes)
            stack_locs.append(locs[c])
            n_nodes += 1

    observed = np.zeros(n_classes, dtype=np.int64)
    while len(stack_ids) > 0:
        node = stack_ids.pop()
        loc = stack_locs.pop()

        # chi-squared score and upper bound of the node
        observed[:] = 0
        for v in loc:
            if v < 0:
                observed[labels[-v - 1]] += 1
        score = _chi_square(observed, y_prob)
        bound = _chi_square_bound(observed, y_prob)
        chi[node] = score

        # NodeStore::insert_node
        threshold = chi[store[store_n - 1]] if store_n == capacity else 0.0
        if not score < threshold and not covered[node]:
            lo = 0
            hi = store_n
            while lo < hi:
                mid = (lo + hi) // 2
                if score > chi[store[mid]]:
                    hi = mid
                else:
                    lo = mid + 1
            for k in range(store_n, lo, -1):
                store[k] = store[k - 1]
            store[lo] = node
            store_n += 1
            selected[node] = True
            while store_n > capacity:
                selected[store[store_n - 1]] = False
                store_n -= 1

        # prune the branch
        threshold = chi[store[store_n - 1]] if store_n == capacity else 0.0
        if bound <= threshold:
            continue

        # expand the node with the next character of each location
        pos_count[:] = 0
        doc_count[:] = 0
        last_doc[:] = -1
        doc = -1
        for v in loc:
            if v < 0:
                doc = -v - 1
            else:
                p = v + 1
                if p < seq_len[doc]:
                    c = _char_at(words, doc, p, word_length)
                    if c != _SPACE:
                        pos_count[c] += 1
                        if last_doc[c] != doc:
                            doc_count[c] += 1
                            last_doc[c] = doc
        if pos_count.sum() == 0:
            continue

        child_locs = List()
        child_chars = List()
        for c in range(256):
            if pos_count[c] > 0:
                child_locs.append(np.empty(pos_count[c] + doc_count[c], dtype=np.int64))
                child_chars.append(c)
        slot = np.full(256, -1, dtype=np.int64)
        for k in range(len(child_chars)):
            slot[child_chars[k]] = k
        fill[:] = 0
        last_doc[:] = -1
        doc = -1
        for v in loc:
            if v < 0:
                doc = -v - 1
            else:
                p = v + 1
                if p < seq_len[doc]:
                    c = _char_at(words, doc, p, word_length)
                    if c != _SPACE:
                        cl = child_locs[slot[c]]
                        if last_doc[c] != doc:
                            cl[fill[c]] = -(doc + 1)
                            fill[c] += 1
                            last_doc[c] = doc
                        cl[fill[c]] = p
                        fill[c] += 1

        check_footprint = selected[node] or covered[node]
        for k in range(len(child_chars)):
            if n_nodes == node_cap:
                node_cap *= 2
                parent = _grow(parent, node_cap)
                last_char = _grow(last_char, node_cap)
                chi = _grow(chi, node_cap)
                selected = _grow(selected, node_cap)
                covered = _grow(covered, node_cap)
            parent[n_nodes] = node
            last_char[n_nodes] = child_chars[k]
            selected[n_nodes] = False
            covered[n_nodes] = check_footprint and _footprint_match(child_locs[k], loc)
            stack_ids.append(n_nodes)
            stack_locs.append(child_locs[k])
            n_nodes += 1

    # ngrams of the stored nodes, in descending chi-squared order
    lengths = np.zeros(store_n, dtype=np.int64)
    for k in range(store_n):
        n = store[k]
        while n >= 0:
            lengths[k] += 1
            n = parent[n]
    out = np.zeros((store_n, max(lengths.max(), 1) if store_n > 0 else 1), np.uint8)
    for k in range(store_n):
        n = store[k]
        j = lengths[k] - 1
        while n >= 0:
            out[k, j] = last_char[n]
            j -= 1
            n = parent[n]
    return out, lengths
