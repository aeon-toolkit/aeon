"""Piecewise Aggregate Approximation Transformer (PAA)."""

__maintainer__ = []

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs


class PAA(BaseCollectionTransformer):
    """
    Piecewise Aggregate Approximation Transformer (PAA).

    (PAA) Piecewise Aggregate Approximation Transformer, as described in [1]. For
    each series reduce the dimensionality to n_segments, where each value is the
    mean of values in the interval.

    Parameters
    ----------
    n_segments : int, default = 8
        Dimension of the transformed data.

    References
    ----------
    [1]  Eamonn Keogh, Kaushik Chakrabarti, Michael Pazzani, and Sharad Mehrotra.
    Dimensionality reduction for fast similarity search in large time series
    databases. Knowledge and information Systems, 3(3), 263-286, 2001.

    Examples
    --------
    >>> from aeon.transformations.collection.dictionary_based import PAA
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> paa = PAA(n_segments=10)
    >>> X_train_paa = paa.fit_transform(X_train)
    >>> X_test_paa = paa.transform(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "fit_is_empty": True,
        "algorithm_type": "dictionary",
    }

    def __init__(self, n_segments=8, n_jobs=1):
        self.n_segments = n_segments
        self.n_jobs = n_jobs

        super().__init__()

    @njit(parallel=True, fastmath=True, cache=True)
    def _transform(self, X, y=None):
        """Transform the input time series to PAA segments.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The input time series
        y : np.ndarray of shape = (n_cases,), default = None
            The labels are not used

        Returns
        -------
        X_paa : np.ndarray of shape = (n_cases, n_channels, n_segments)
                The output of the PAA transformation
        """
        prev_threads = get_num_threads()
        _n_jobs = check_n_jobs(self.n_jobs)

        set_num_threads(_n_jobs)

        n_segments = self.n_segments
        n_cases, n_channels, n_timepoints = X.shape
        X_paa = np.zeros((n_cases, n_channels, n_segments), dtype=np.float64)

        # find the number of elements in each segment, then mean of these
        # elements will be stored in place of multiple time-points.
        segment_length = n_timepoints / n_segments

        for i_x in prange(n_cases):
            for i_c in range(n_channels):
                for seg in range(n_segments):
                    start_idx = int(seg * segment_length)

                    # Calculate end index, handling the last segment
                    if seg == n_segments - 1:
                        end_idx = n_timepoints
                    else:
                        end_idx = int((seg + 1) * segment_length)

                    # Compute mean of segment

                    if end_idx > start_idx:
                        segment_sum = 0.0
                        for t in range(start_idx, end_idx):
                            segment_sum += X[i_x, i_c, t]
                            X_paa[i_x, i_c, seg] = segment_sum / (end_idx - start_idx)

        set_num_threads(prev_threads)
        return X_paa

    def inverse_paa(self, X, original_length):
        """Produce the inverse PAA transformation.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_segments)
            The output of the PAA transformation
        original_length : int
            The original length of the series.

        Returns
        -------
        np.ndarray
            (n_cases, n_channels, n_timepoints) the inverse of paa transform.
        """
        if original_length % self.n_segments == 0:
            return np.repeat(X, repeats=int(original_length / self.n_segments), axis=-1)

        else:
            n_samples, n_channels, _ = X.shape
            X_inverse_paa = np.zeros(shape=(n_samples, n_channels, original_length))

            all_indices = np.arange(original_length)
            split_segments = np.array_split(all_indices, self.n_segments)

            for _s, segment in enumerate(split_segments):
                X_inverse_paa[:, :, segment] = np.repeat(
                    X[:, :, [_s]], repeats=len(segment), axis=-1
                )

            return X_inverse_paa

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
        params = {"n_segments": 10}
        return params
