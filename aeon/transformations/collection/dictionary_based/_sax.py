"""Symbolic Aggregate approXimation (SAX) transformer."""

__maintainer__ = []
__all__ = ["SAX", "_invert_sax_symbols"]

import numpy as np
import scipy.stats
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.dictionary_based import PAA
from aeon.utils.validation import check_n_jobs


class SAX(BaseCollectionTransformer):
    """Symbolic Aggregate approXimation (SAX) transformer.

    as described in
    Jessica Lin, Eamonn Keogh, Li Wei and Stefano Lonardi,
    "Experiencing SAX: a novel symbolic representation of time series"
    Data Mining and Knowledge Discovery, 15(2):107-144

    Parameters
    ----------
    n_segments : int, default = 8,
        number of segments for the PAA, each segment is represented
        by a symbol
    alphabet_size : int, default = 4,
        size of the alphabet to be used to create the bag of words
    alphabet : list, default = None,
        The alphabet to be used to create the bag of words, if this
        parameter is None then the alphabet are simply the inteeger
        values from 0 to alphabet_size - 1, if this parameter is not None
        then the length of the alphabet should be equal to the
        alphabet_size parameter.
    distribution : str, default = "Gaussian",
        options={"Gaussian"}
        the distribution function to use when generating the
        alphabet. Currently only Gaussian is supported.
    distribution_params : dict, default = None,
        the parameters of the used distribution, if the used
        distribution is "Gaussian" and this parameter is None
        then the default setup is {"scale" : 1.0}
    znormalized : bool, default = True,
        this parameter is set to True when the input time series
        are assume to be z-normalized, i.e. the mean of each
        time series should be 0 and the standard deviation should be
        equal to 1. If this parameter is set to False, the z-normalization
        is applied before the transformation.
    window_size : int, default = None,
        The size of the sliding window to use when transforming the time series,
        if this parameter is None then the whole time series is used to
        produce the SAX transformation, if this parameter is not None then
        the SAX transformation is applied to each sliding window of the time series.
    stride : int, default = 1,
        The stride to use when applying the sliding window, this parameter is
        only used when the window_size parameter is not None.
    n_jobs : int, default = 1,
        The number of jobs to run in parallel for both `fit` and `transform`.

    Notes
    -----
    This implementation is based on the one done by tslearn [1]

    References
    ----------
    .. [1] https://github.com/tslearn-team/tslearn/blob/fa40028/tslearn/
       piecewise/piecewise.py#L261-L501

    Examples
    --------
    >>> from aeon.transformations.collection.dictionary_based import SAX
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> sax = SAX(n_segments=10, alphabet_size=8)
    >>> X_train = sax.fit_transform(X_train)
    >>> X_test = sax.fit_transform(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "fit_is_empty": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        n_segments: int = 8,
        alphabet_size: int = 4,
        alphabet: list = None,
        distribution: str = "Gaussian",
        distribution_params: dict = None,
        znormalized: bool = True,
        window_size: int = None,
        stride: int = 1,
        n_jobs: int = 1,
    ):
        self.n_segments = n_segments

        self.alphabet_size = alphabet_size
        self.alphabet = alphabet
        assert (
            self.alphabet is None or len(self.alphabet) == self.alphabet_size
        ), "The length of the alphabet should be equal to the alphabet_size parameter"

        self.distribution = distribution
        self.n_jobs = n_jobs
        self.distribution_params = distribution_params
        self.znormalized = znormalized

        self.window_size = window_size
        self.stride = stride

        if self.distribution == "Gaussian":
            self.distribution_params_ = (
                dict(scale=1.0)
                if self.distribution_params is None
                else self.distribution_params
            )

        else:
            raise NotImplementedError(self.distribution, "still not added")

        self.breakpoints, self.breakpoints_mid = self._generate_breakpoints(
            alphabet_size=self.alphabet_size,
            distribution=self.distribution,
            distribution_params=self.distribution_params_,
        )

        super().__init__()

    def _validate_window_parameters(self, n_timepoints):
        if self.window_size is None:
            return

        if not isinstance(self.window_size, int):
            raise TypeError("window_size must be an integer or None")

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if self.window_size > n_timepoints:
            raise ValueError("window_size cannot be greater than the series length")

        if self.window_size < self.n_segments:
            raise ValueError("window_size must be at least as large as n_segments")

        if not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError("stride must be a positive integer")

    def _get_paa(self, X):
        """Transform the input time series to PAA segments.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The input time series

        Returns
        -------
        X_paa : np.ndarray of shape = (n_cases, n_channels, n_segments)
            The output of the PAA transformation
        """
        if not self.znormalized:
            means = np.mean(X, axis=-1, keepdims=True)
            stds = np.std(X, axis=-1, keepdims=True)
            stds[stds == 0] = 1.0
            X = (X - means) / (stds)

        paa = PAA(n_segments=self.n_segments, n_jobs=self.n_jobs)
        X_paa = paa.fit_transform(X=X)

        return X_paa

    def _transform(self, X, y=None):
        """Transform the input time series to SAX symbols.

        This function will transform the input time series into a bag of
        symbols. These symbols are represented as integer values pointing
        to the indices of the breakpoint in the alphabet for each
        segment produced by PAA.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The input time series
        y : np.ndarray of shape = (n_cases,), default = None
            The labels are not used

        Returns
        -------
        sax_symbols : np.ndarray of shape = (n_cases, n_channels, n_segments)
            The output of the SAX transformation
        """
        n_cases, n_channels, n_timepoints = X.shape

        if self.window_size is None:
            X_paa = self._get_paa(X)
            return self._get_sax_symbols(X_paa)

        self._validate_window_parameters(n_timepoints)

        X_windows = _extract_windows(
            X,
            window_size=self.window_size,
            stride=self.stride,
        )

        n_windows = X_windows.shape[2]

        # Convert:
        # (cases, channels, windows, window_size)
        # into:
        # (cases * windows, channels, window_size)
        X_windows_3d = X_windows.transpose(0, 2, 1, 3).reshape(
            n_cases * n_windows,
            n_channels,
            self.window_size,
        )

        X_paa = self._get_paa(X_windows_3d)
        sax_symbols = self._get_sax_symbols(X_paa)

        # Convert:
        # (cases * windows, channels, n_segments)
        # into:
        # (cases, channels, windows, n_segments)
        sax_symbols = sax_symbols.reshape(
            n_cases,
            n_windows,
            n_channels,
            self.n_segments,
        ).transpose(0, 2, 1, 3)

        return sax_symbols

    def _get_sax_symbols(self, X_paa):
        """Produce the SAX transformation.

        Parameters
        ----------
        X_paa : np.ndarray of shape = (n_cases, n_channels, n_segments)
            The output of the PAA transformation

        Returns
        -------
        sax_symbols : np.ndarray of shape = (n_cases, n_channels, n_segments)
            The output of the SAX transformation using np.digitize
        """
        prev_threads = get_num_threads()
        _n_jobs = check_n_jobs(self.n_jobs)

        set_num_threads(_n_jobs)
        sax_symbols = _parallel_get_sax_symbols(
            X_paa, breakpoints=self.breakpoints, right=False, alphabet=self.alphabet
        )

        set_num_threads(prev_threads)

        return sax_symbols

    def inverse_sax(self, X, original_length=None, y=None):
        """Reconstruct time series from SAX symbols.

        Supports both standard and windowed SAX output.

        Parameters
        ----------
        X : np.ndarray
            Standard SAX shape:
                (n_cases, n_channels, n_segments)

            Windowed SAX shape:
                (n_cases, n_channels, n_windows, n_segments)

        original_length : int, optional
            Required for standard SAX.

            For windowed SAX, this is the desired reconstructed series length.
            If omitted, the covered length is inferred from the number of
            windows, window size and stride.

        Returns
        -------
        np.ndarray
            Reconstructed series with shape:
                (n_cases, n_channels, n_timepoints)
        """
        X = np.asarray(X)

        if self.alphabet is None:
            sax_indices = X.astype(np.intp, copy=False)
        else:
            alphabet = np.asarray(self.alphabet)

            if len(np.unique(alphabet)) != len(alphabet):
                raise ValueError(
                    "alphabet values must be unique to perform inverse_sax"
                )

            sax_indices = self._alphabet_to_indices(X, alphabet)

        previous_threads = get_num_threads()
        n_jobs = check_n_jobs(self.n_jobs)

        try:
            set_num_threads(n_jobs)

            # Standard SAX output:
            # (n_cases, n_channels, n_segments)
            if sax_indices.ndim == 3:
                if original_length is None:
                    raise ValueError(
                        "original_length is required for non-windowed inverse SAX"
                    )

                return _invert_sax_symbols(
                    sax_symbols=sax_indices,
                    n_timepoints=original_length,
                    breakpoints_mid=self.breakpoints_mid,
                )

            # Windowed SAX output:
            # (n_cases, n_channels, n_windows, n_segments)
            if sax_indices.ndim == 4:
                if self.window_size is None:
                    raise ValueError(
                        "A 4D SAX array requires window_size to be configured"
                    )

                effective_stride = (
                    self.window_size if self.stride is None else self.stride
                )

                n_windows = sax_indices.shape[2]

                covered_length = (n_windows - 1) * effective_stride + self.window_size

                if original_length is None:
                    original_length = covered_length

                if original_length < covered_length:
                    raise ValueError(
                        "original_length cannot be smaller than the length "
                        "covered by the SAX windows"
                    )

                return _invert_windowed_sax_symbols(
                    sax_symbols=sax_indices,
                    original_length=original_length,
                    window_size=self.window_size,
                    stride=effective_stride,
                    breakpoints_mid=self.breakpoints_mid,
                )

            raise ValueError(
                "X must have shape "
                "(n_cases, n_channels, n_segments) or "
                "(n_cases, n_channels, n_windows, n_segments)"
            )

        finally:
            set_num_threads(previous_threads)

    def _generate_breakpoints(
        self, alphabet_size, distribution="Gaussian", distribution_params=None
    ):
        """Generate the breakpoints following a probability distribution.

        Parameters
        ----------
        alphabet_size : int
            The size of the alphabet, the number of breakpints is alphabet_size -1
        distribution : str, default = "Gaussian"
            The name of the distribution to follow when generating the breakpoints
        distribution_params : dict, default = None,
            The parameters of the chosen distribution, if the distribution is
            chosen as Gaussian and this is left default to None then the params
            used are "scale=1.0"

        Returns
        -------
        breakpoints : np.ndarray of shape = (alphabet_size-1,)
            The breakpoints to be used to generate the SAX symbols
        breakpoints_mid : np.ndarray of shape = (alphabet_size,)
            The middle breakpoints for each breakpoint interval used to produce
            the inverse of SAX transformation
        """
        if distribution == "Gaussian":
            breakpoints = scipy.stats.norm.ppf(
                np.arange(1, alphabet_size, dtype=np.float64) / alphabet_size,
                scale=distribution_params["scale"],
            )

            breakpoints_mid = scipy.stats.norm.ppf(
                np.arange(1, 2 * alphabet_size, 2, dtype=np.float64)
                / (2 * self.alphabet_size),
                scale=distribution_params["scale"],
            )

        return breakpoints, breakpoints_mid

    def _alphabet_to_indices(self, sax_symbols, alphabet):
        """Convert custom SAX symbols back to integer alphabet positions."""
        lookup = {symbol: index for index, symbol in enumerate(alphabet.tolist())}

        flat_symbols = sax_symbols.ravel()
        flat_indices = np.empty(flat_symbols.size, dtype=np.intp)

        for i, symbol in enumerate(flat_symbols):
            try:
                flat_indices[i] = lookup[symbol]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown SAX symbol {symbol!r}. "
                    f"Expected one of {alphabet.tolist()}."
                ) from exc

        return flat_indices.reshape(sax_symbols.shape)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {"n_segments": 10, "alphabet_size": 8}
        return params


@njit(cache=True, parallel=True)
def _extract_windows(X, window_size, stride):
    n_cases, n_channels, n_timepoints = X.shape

    n_windows = 1 + (n_timepoints - window_size) // stride

    windows = np.empty(
        (n_cases, n_channels, n_windows, window_size),
        dtype=X.dtype,
    )

    for i in prange(n_cases):
        for c in range(n_channels):
            for w in range(n_windows):
                start = w * stride
                stop = start + window_size
                windows[i, c, w, :] = X[i, c, start:stop]

    return windows


@njit(parallel=True, fastmath=True)
def _invert_sax_symbols(sax_symbols, n_timepoints, breakpoints_mid):
    """Reconstruct the original time series using a Gaussian estimation.

    In other words, try to inverse the SAX transformation.

    Parameters
    ----------
    sax_symbols : np.ndarray(n_cases, n_channels, n_segments)
        The sax output transformation
    n_timepoints : int
        The original time series length
    breakpoints_mid : np.ndarray(alphabet_size)
        The Gaussian estimation of the value for each breakpoint interval

    Returns
    -------
    sax_inverse : np.ndarray(n_cases, n_channels, n_timepoints)
        The inverse of sax transform
    """
    n_cases, n_channels, n_segments = sax_symbols.shape

    result = np.empty(
        (n_cases, n_channels, n_timepoints),
        dtype=breakpoints_mid.dtype,
    )

    for i in prange(n_cases):
        for c in range(n_channels):
            for t in range(n_timepoints):
                segment_index = min(
                    (t * n_segments) // n_timepoints,
                    n_segments - 1,
                )

                symbol_index = sax_symbols[
                    i,
                    c,
                    segment_index,
                ]

                result[i, c, t] = breakpoints_mid[symbol_index]

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _invert_windowed_sax_symbols(
    sax_symbols,
    original_length,
    window_size,
    stride,
    breakpoints_mid,
):
    """Invert windowed SAX using overlap averaging."""
    n_cases, n_channels, n_windows, n_segments = sax_symbols.shape

    reconstructed = np.zeros(
        (n_cases, n_channels, original_length),
        dtype=breakpoints_mid.dtype,
    )

    counts = np.zeros(
        (n_cases, n_channels, original_length),
        dtype=np.intp,
    )

    for i in prange(n_cases):
        for c in range(n_channels):
            for w in range(n_windows):
                window_start = w * stride

                for local_t in range(window_size):
                    global_t = window_start + local_t

                    if global_t >= original_length:
                        continue

                    segment_index = min(
                        (local_t * n_segments) // window_size,
                        n_segments - 1,
                    )

                    symbol_index = sax_symbols[
                        i,
                        c,
                        w,
                        segment_index,
                    ]

                    reconstructed[i, c, global_t] += breakpoints_mid[symbol_index]

                    counts[i, c, global_t] += 1

            for t in range(original_length):
                if counts[i, c, t] > 0:
                    reconstructed[i, c, t] /= counts[i, c, t]
                else:
                    reconstructed[i, c, t] = np.nan

    return reconstructed


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_get_sax_indices(x, breakpoints, right=False):
    """Parallel version using np.digitize within prange loop."""
    n_samples, n_channels, _ = x.shape
    result = np.empty_like(x, dtype=np.intp)

    for i in prange(n_samples):
        for c in range(n_channels):
            _indices = np.digitize(x[i, c, :], breakpoints, right=right)

            result[i, c, :] = _indices

    return result


def _parallel_get_sax_symbols(x, breakpoints, right=False, alphabet=None):
    breakpoints = np.asarray(breakpoints)

    indices = _parallel_get_sax_indices(
        x=x,
        breakpoints=breakpoints,
        right=right,
    )

    if alphabet is None:
        return indices

    alphabet = np.asarray(alphabet)

    return alphabet[indices]
