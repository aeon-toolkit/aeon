"""BinSeg (Binary segmentation) Segmenter."""

__maintainer__ = []
__all__ = ["BinSegmenter"]

import numpy as np
import pandas as pd

from aeon.segmentation.base import BaseSegmenter


class BinSegmenter(BaseSegmenter):
    """BinSeg (Binary Segmentation) Segmenter.

    From the Ruptures documentation:
    "Binary change point detection is used to perform fast signal segmentation and is
    implemented in Binseg. It is a sequential approach: first, one change point is
    detected in the complete input signal, then series is split around this change point
    then the operation is repeated on the two resulting sub-signals. For a theoretical
    and algorithmic analysis of Binseg, see for instance [1] and [2].
    The benefits of binary segmentation includes low complexity (of the order of , where
    is the number of samples and  the complexity of calling the considered cost function
    on one sub-signal), the fact that it can extend any single change point detection
    method to detect multiple changes points and that it can work whether the number of
    regimes is known beforehand or not."

    Parameters
    ----------
    n_cps : int, default = 1
        The number of change points to search.
    model : str, default = "l2"
        Segment model to use. Options are "l1", "l2", "rbf", etc.
        (see ruptures documentation for available models).
    min_size : int, default = 2,
        Minimum segment length. Defaults to 2 samples.
    jump : int, default = 5,
        Subsample (one every jump points). Defaults to 5.

    References
    ----------
    .. [1] Bai, J. (1997). Estimating multiple breaks one at a time.
    Econometric Theory, 13(3), 315–352.

    .. [2] Fryzlewicz, P. (2014). Wild binary segmentation for multiple
    change-point detection. The Annals of Statistics, 42(6), 2243–2281.

    Examples
    --------
    >>> from aeon.segmentation import BinSegmenter
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation()
    >>> binseg = BinSegmenter(n_cps=1)  # doctest: +SKIP
    >>> found_cps = binseg.fit_predict(X)  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
        "python_dependencies": "ruptures",
    }

    def __init__(self, n_cps=1, model="l2", min_size=2, jump=5):
        self.n_cps = n_cps
        self.model = model
        self.min_size = min_size
        self.jump = jump
        super().__init__(n_segments=n_cps + 1, axis=1)

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
        X = X.squeeze()
        found_cps = self._run_binseg(X)
        return found_cps

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {}

    def _run_binseg(self, X):
        import ruptures as rpt

        binseg = rpt.Binseg(
            model=self.model, min_size=self.min_size, jump=self.jump
        ).fit(X)
        found_cps = np.array(binseg.predict(n_bkps=self.n_cps)[:-1], dtype=np.int64)

        return found_cps

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
           Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps] The found change points found

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
        return {"n_cps": 1}
