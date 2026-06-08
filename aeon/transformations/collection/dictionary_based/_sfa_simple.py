"""Symbolic Fourier Approximation (SFA) Transformer.

Configurable SFA transform for discretising time series into words.
"""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["SFA_Simple"]

import numpy as np
from sklearn.utils import check_random_state

from aeon.transformations.collection import BaseCollectionTransformer


class SFA_Simple(BaseCollectionTransformer):
    """Symbolic Fourier Approximation (SFA) [1]_ Transformer.

    For each series, shorten the series with DFT
    discretise the shortened series into bins set by MFC
    form a word from these discrete values.

    Parameters
    ----------
    word_length : int, default = 8,
        The number of segments shortened to (using DFT).
    alphabet_size : int, default = 4
        The size of the alphabet to be used to create
        the bag of words.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    References
    ----------
    .. [1] Schäfer, Patrick, and Mikael Högqvist. "SFA: a symbolic fourier approximation
    and  index for similarity search in high dimensional datasets." Proceedings of the
    15th international conference on extending database technology. 2012.
    """

    _tags = {
        "capability:multivariate": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        random_state=None,
    ):
        self.breakpoints = []
        self.bins_ = []

        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.random_state = random_state

        self.n_cases = 0
        self.n_channels = 0
        self.n_timepoints = 0

        super().__init__()

    def _fit(self, X, y=None):
        """Calculate word breakpoints using MCB or IGB.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The input time series.
        y : np.ndarray of shape = (n_cases,), default = None
            The labels are not used.

        Returns
        -------
        self: object
        """
        self.n_cases, self.n_channels, self.n_timepoints = X.shape

        # though no randomness introduced yet
        rng = check_random_state(self.random_state)
        self.random_state_ = rng.randint(0, np.iinfo(np.int32).max)
        np.random.seed(self.random_state_)

        for c in range(self.n_channels):

            X_c = X[:, c, :]

            dft_features = np.array([self._compute_dft(x_c) for x_c in X_c])

            bins = self._uniform_binning(dft_features)
            self.bins_.append(bins)

    def _compute_dft(self, x):
        coeffs = np.fft.rfft(x)
        dft = np.empty(self.word_length, dtype=np.float32)

        num_coeffs = self.word_length // 2
        assert num_coeffs <= len(coeffs)

        reals = np.real(coeffs)
        imags = np.imag(coeffs)

        if self.word_length % 2 == 1:
            dft[0::2] = reals[: 1 + np.uint32(self.word_length / 2)]
            dft[1::2] = imags[
                : (self.word_length - 1 - np.uint32(self.word_length / 2))
            ]
        else:
            dft[0::2] = reals[: np.uint32(self.word_length / 2)]
            dft[1::2] = imags[: (self.word_length - np.uint32(self.word_length / 2))]

        return dft

    def _uniform_binning(self, values):
        bins = []
        self.breakpoints = list(range(self.alphabet_size))

        for t in range(self.word_length):
            values_t = values[:, t]
            min_val_t, max_val_t = np.min(values_t), np.max(values_t)

            if min_val_t == max_val_t:
                edges = [-np.inf] + [np.inf] * (self.alphabet_size - 1)
            else:
                edges = list(
                    np.linspace(min_val_t, max_val_t, self.alphabet_size + 1)[1:-1]
                )

            bins.append(edges)
        return bins

    def _discretize(self, values, bins):
        word = []
        for i, val in enumerate(values):
            edges, symbols = bins[i], self.breakpoints
            bin_idx = np.searchsorted(edges, val, side="right")
            word.append(symbols[bin_idx])
        return word

    def _transform(self, X, y=None):
        """Transform the input time series to SFA symbols.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The input time series.
        y : np.ndarray of shape = (n_cases,), default = None
            The labels are not used.

        Returns
        -------
        sfa_symbols : np.ndarray of shape = (n_cases, n_channels, word_length)
            The output of the SFA transformation.
        """
        n_cases_ = len(X)
        sfa_symbols = np.zeros(
            shape=(n_cases_, self.n_channels, self.word_length), dtype=np.int32
        )

        for c in range(self.n_channels):
            X_c = X[:, c, :]
            dft_features = np.array([self._compute_dft(x_c) for x_c in X_c])
            bins_c = self.bins_[c]

            for n in range(n_cases_):
                sfa_symbols[n, c] = self._discretize(dft_features[n], bins_c)

        return sfa_symbols
