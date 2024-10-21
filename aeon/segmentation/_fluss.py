"""FLUSS (Fast Low-cost Unipotent Semantic Segmentation) Segmenter."""

__maintainer__ = []
__all__ = ["FLUSSSegmenter"]

import numpy as np
import pandas as pd

from aeon.segmentation.base import BaseSegmenter


class FLUSSSegmenter(BaseSegmenter):
    """FLUSS (Fast Low-cost Unipotent Semantic Segmentation) Segmenter.

    FLOSS [1]_ FLUSS is a domain-agnostic online semantic segmentation method that
    operates on the assumption that a low number of arcs crossing a given index point
    indicates a high probability of a semantic change.

    Parameters
    ----------
    period_length : int, default = 10
        Size of window for sliding, based on the period length of the data.
    n_regimes : int, default = 2
        The number of regimes to search (equal to change points - 1).
    exclusion_factor : int, default 5
        The multiplying factor for the regime exclusion zone

    References
    ----------
    .. [1] Gharghabi S, Ding Y, Yeh C-CM, Kamgar K, Ulanova L, Keogh E. Matrix
    Profile VIII: Domain Agnostic Online Semantic Segmentation at Superhuman Performance
    Levels. In: 2017 IEEE International Conference on Data Mining (ICDM). IEEE; 2017.
    p. 117-26.

    Examples
    --------
    >>> from aeon.segmentation import FLUSSSegmenter
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation()
    >>> fluss = FLUSSSegmenter(period_length=10, n_regimes=2)  # doctest: +SKIP
    >>> found_cps = fluss.fit_predict(X)  # doctest: +SKIP
    >>> profiles = fluss.profiles  # doctest: +SKIP
    >>> scores = fluss.scores  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
        "python_dependencies": "stumpy",
    }

    def __init__(self, period_length=10, n_regimes=2, exclusion_factor=5):
        self.period_length = int(period_length)
        self.n_regimes = n_regimes
        self.exclusion_factor = exclusion_factor
        super().__init__(n_segments=n_regimes, axis=1)

    def _predict(self, X: np.ndarray):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : np.ndarray
            1D time series to be segmented.

        Returns
        -------
        list
            List of change points found in X.
        """
        if self.n_regimes < 2:
            raise ValueError(
                "The number of regimes must be set to an integer greater than 1"
            )

        X = X.squeeze()
        self.found_cps, self.profiles, self.scores = self._run_fluss(X)
        return self.found_cps

    def predict_scores(self, X):
        """Return scores in FLUSS's profile for each annotation.

        Parameters
        ----------
        np.ndarray
            1D time series to be segmented.

        Returns
        -------
        np.ndarray
            Scores for sequence X
        """
        self.found_cps, self.profiles, self.scores = self._run_fluss(X)
        return self.scores

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {"profile": self.profile}

    def _run_fluss(self, X):
        import stumpy

        mp = stumpy.stump(X, m=self.period_length)
        self.profile, self.found_cps = stumpy.fluss(
            mp[:, 1],
            L=self.period_length,
            excl_factor=self.exclusion_factor,
            n_regimes=self.n_regimes,
        )
        self.scores = self.profile[self.found_cps]

        return self.found_cps, self.profile, self.scores

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
            Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps]
            The found change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change points

        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)

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
        return {"period_length": 5, "n_regimes": 2}
