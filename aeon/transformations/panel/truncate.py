# -*- coding: utf-8 -*-
"""Truncation transformer - truncate unequal length panels to lower/upper bounds."""
import numpy as np

from aeon.transformations.base import BaseTransformer

__all__ = ["TruncationTransformer"]
__author__ = ["abostrom", "TonyBagnall"]


class TruncationTransformer(BaseTransformer):
    """Truncate unequal length time series to a lower bounds.

    Truncates all series in panel between lower/upper range bounds. This transformer
    assumes that all series have the same number of channels (dimensions) and
    that all channels in a single series are the same length.

    Parameters
    ----------
    truncated_length : int, optional (default=None) bottom range of the values to
                truncate can also be used to truncate to a specific length
                if None, will find the shortest sequence and use instead.

    Example
    -------

    """

    _tags = {
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": ["np-list", "numpy3D"],
        "y_inner_mtype": "None",
        "fit_is_empty": False,
        "capability:unequal_length:removes": True,
    }

    def __init__(self, truncated_length=None):
        self.truncated_length = truncated_length
        super(TruncationTransformer, self).__init__(_output_convert=False)

    @staticmethod
    def _get_min_length(X):
        min_length = X[0].shape[1]
        for x in X:
            if x.shape[1] < min_length:
                min_length = x.shape[1]

        return min_length

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series or
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        # If lower is none, set to the minimum length in X
        min_length = self._get_min_length(X)
        if self.truncated_length is None:
            self.truncated_length_ = min_length
        elif min_length < self.truncated_length:
            self.truncated_length_ = min_length
        else:
            self.truncated_length_ = self.truncated_length

        return self

    def _transform(self, X, y=None):
        """Truncate X and return a transformed version.

        Parameters
        ----------
        X : list of [n_cases] 2D np.ndarray shape (n_channels, length_i)
            where length_i can vary between time series.
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : numpy3D array (n_cases, n_channels, self.truncated_length_)
            truncated time series from X.
        """
        min_length = self._get_min_length(X)
        if min_length < self.truncated_length_:
            raise ValueError(
                "Error: min_length of series \
                    is less than the one found when fit or set."
            )
        Xt = []
        for x in X:
            Xt.append(x[:, : self.truncated_length_])
        Xt = np.array(Xt)
        return Xt

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
        params = {"lower": 5}
        return params
