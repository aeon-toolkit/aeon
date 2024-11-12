"""Piecewise Aggregate Approximation Transformer (PAA)."""

__maintainer__ = []

import numpy as np

from aeon.transformations.collection import BaseCollectionTransformer


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

    def __init__(self, n_segments=8):
        self.n_segments = n_segments

        super().__init__()

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
        length_TS = int(X.shape[-1])
        all_indices = np.arange(length_TS)

        # The following will include the left out indices
        # For instance if the length of the TS is 10 and the number
        # of segments is 3, the indices will be [0:3], [3:6] and [6:10]
        # so 3 segments, two of length 3 and one of length 4
        split_segments = np.array_split(all_indices, self.n_segments)

        # If the series length is divisible by the number of segments
        # then the transformation can be done in one line
        # If not, a for loop is needed only on the segments while
        # parallelizing the transformation

        if length_TS % self.n_segments == 0:
            X_paa = X[:, :, split_segments].mean(axis=-1)
            return X_paa

        else:
            n_samples, n_channels, _ = X.shape
            X_paa = np.zeros(shape=(n_samples, n_channels, self.n_segments))

            for _s, segment in enumerate(split_segments):
                if X[:, :, segment].shape[-1] > 0:  # avoids mean of empty slice error
                    X_paa[:, :, _s] = X[:, :, segment].mean(axis=-1)

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
