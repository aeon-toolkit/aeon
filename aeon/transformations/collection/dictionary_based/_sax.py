"""Symbolic Aggregate approXimation (SAX) transformer."""

__maintainer__ = []
__all__ = ["SAX", "_invert_sax_symbols"]

import numpy as np
import scipy.stats
from numba import njit, prange

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.dictionary_based import PAA


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
        "fit_is_empty": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        n_segments=8,
        alphabet_size=4,
        distribution="Gaussian",
        distribution_params=None,
        znormalized=True,
    ):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.distribution = distribution

        self.distribution_params = distribution_params
        self.znormalized = znormalized

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
            # Safe, if std is 0
            X = (X - np.mean(X, axis=-1, keepdims=True)) / (
                np.std(X, axis=-1, keepdims=True) + 1e-8
            )

            # Non-Safe is std is 0
            # X = scipy.stats.zscore(X, axis=-1)

        paa = PAA(n_segments=self.n_segments)
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
        X_paa = self._get_paa(X=X)
        sax_symbols = self._get_sax_symbols(X_paa=X_paa)
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
        sax_symbols = np.digitize(x=X_paa, bins=self.breakpoints)
        return sax_symbols

    def inverse_sax(self, X, original_length, y=None):
        """Produce the inverse SAX transformation.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_segments)
            The output of the SAX transformation
        y : np.ndarray of shape = (n_cases,), default = None
            The labels are not used

        Returns
        -------
        sax_inverse : np.ndarray(n_cases, n_channels, n_timepoints)
            The inverse of sax transform
        """
        sax_inverse = _invert_sax_symbols(
            sax_symbols=X,
            n_timepoints=original_length,
            breakpoints_mid=self.breakpoints_mid,
        )

        return sax_inverse

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
    n_samples, n_channels, sax_length = sax_symbols.shape

    segment_length = int(n_timepoints / sax_length)
    sax_inverse = np.zeros((n_samples, n_channels, n_timepoints))

    for i in prange(n_samples):
        for c in prange(n_channels):
            for _current_sax_index in prange(sax_length):
                start_index = _current_sax_index * segment_length
                stop_index = start_index + segment_length

                sax_inverse[i, :, start_index:stop_index] = breakpoints_mid[
                    sax_symbols[i, c, _current_sax_index]
                ]

    return sax_inverse
