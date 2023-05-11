# -*- coding: utf-8 -*-
"""Symbolic Aggregate approXimation (SAX) transformer."""

__author__ = ["MatthewMiddlehurst", "hadifawaz1999"]
__all__ = ["SAX"]

import numpy as np
import scipy.stats
from numba import int64, njit, prange

from aeon.transformations.base import BaseTransformer
from aeon.transformations.panel.dictionary_based import PAA


class SAX(BaseTransformer):
    """Symbolic Aggregate approXimation (SAX) transformer.

    as described in
    Jessica Lin, Eamonn Keogh, Li Wei and Stefano Lonardi,
    "Experiencing SAX: a novel symbolic representation of time series"
    Data Mining and Knowledge Discovery, 15(2):107-144

    Parameters
    ----------
    n_segments : int, default = 8,
        number of segments for the PAA, each segement is represented
        by a symbol
    alphabet_size : int, default = 4,
        size of the alphabet to be used to create the bag of words
    distribution : str, default = "Gaussian",
        the distribution function to use when generating the
        alphabet
    distribution_params : dict, default = None,
        the parameters of the used distribution, if the used
        distribution is "Gaussian" and this parameter is None
        then the default setup is {"scale" : 1.0}

    Notes
    -----
    This implementation is based on the one done by tslearn [1]

    References
    ----------
    .. [1] https://github.com/tslearn-team/tslearn/blob/fa40028/tslearn/
       piecewise/piecewise.py#L261-L501

    Examples
    --------
    >>> from aeon.transformations.panel.dictionary_based import SAX
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> sax = SAX(n_segments=10, alphabet_size=8)
    >>> X_train = sax.fit_transform(X_train)
    >>> X_test = sax.fit_transform(X_test)
    """

    _tags = {
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_segments=8,
        alphabet_size=4,
        distribution="Gaussian",
        distribution_params=None,
    ):
        self.n_segments = n_segments
        self.alphabet_size = alphabet_size
        self.distribution = distribution

        if self.distribution == "Gaussian":
            self.distribution_params = (
                dict(scale=1.0) if distribution_params is None else distribution_params
            )

        else:
            raise NotImplementedError("still not added")

        super(SAX, self).__init__()

    def _transform(self, X, y=None):
        _, _, series_length = X.shape

        paa = PAA(n_intervals=self.n_segments)
        X_paa = paa.fit_transform(X=X)

        self.breakpoints, self.breakpoints_mid = self._generate_breakpoints(
            alphabet_size=self.alphabet_size,
            distribution=self.distribution,
            distribution_params=self.distribution_params,
        )

        return _inverse_sax_symbols(
            _get_sax_symbols(X_paa=X_paa, breakpoints=self.breakpoints),
            series_length=series_length,
            breakpoints_mid=self.breakpoints_mid,
        )

    def _generate_breakpoints(
        self, alphabet_size, distribution="Gaussian", distribution_params=None
    ):
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
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"n_segments": 10, "alphabet_size": 8}
        return params


@njit("int64[:,:,:](float64[:,:,:],float64[:])", fastmath=True)
def _get_sax_symbols(X_paa, breakpoints):
    sax_symbols = np.zeros(X_paa.shape, dtype=int64) - 1
    alphabet_size = breakpoints.shape[0] + 1

    for i_bp, bp in enumerate(breakpoints):
        indices = np.logical_and(sax_symbols < 0, X_paa < bp)
        for i in prange(indices.shape[0]):
            for j in prange(indices.shape[1]):
                for k in prange(indices.shape[2]):
                    if indices[i, j, k]:
                        sax_symbols[i, j, k] = i_bp

    for i in prange(sax_symbols.shape[0]):
        for j in prange(sax_symbols.shape[1]):
            for k in prange(sax_symbols.shape[2]):
                if sax_symbols[i, j, k] < 0:
                    sax_symbols[i, j, k] = alphabet_size - 1
    # the -1 is because breakpoints have self.alphabet_size - 1 elements

    return sax_symbols


@njit("(float64[:,:,:])(int64[:,:,:],float64[:],int32)", fastmath=True)
def _inverse_sax_symbols(sax_symbols, breakpoints_mid, series_length):
    n_samples, n_channels, sax_length = sax_symbols.shape

    segment_length = int(series_length / sax_length)
    output_sax = np.zeros((n_samples, n_channels, series_length))

    for i in prange(n_samples):
        for _current_sax_index in prange(sax_length):
            start_index = _current_sax_index * segment_length
            stop_index = start_index + segment_length

            for d in prange(n_channels):
                for _index in prange(start_index, stop_index):
                    output_sax[i, d, _index] = breakpoints_mid[
                        sax_symbols[i, d, _current_sax_index]
                    ]

    return output_sax
