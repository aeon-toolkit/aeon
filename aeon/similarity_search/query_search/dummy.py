"""Dummy similarity seach estimator."""

__maintainer__ = ["baraline"]

import numpy as np

from aeon.similarity_search.query_search.base import BaseQuerySearch


class DummyQuerySearch(BaseQuerySearch):
    """
    DummySimilaritySearch estimator, will return the best match to the query.

    Parameters
    ----------
    distance : str, default="euclidean"
        Name of the distance function to use. A list of valid strings can be found in
        the documentation for :func:`aeon.distances.get_distance_function`.
        If a callable is passed it must either be a python function or numba function
        with nopython=True, that takes two 1d numpy arrays as input and returns a float.
    distance_args : dict, default=None
        Optional keyword arguments for the distance function.
    normalize : bool, default=False
        Whether the distance function should be z-normalized.
    store_distance_profiles : bool, default=False.
        Whether to store the computed distance profile in the attribute
        "distance_profiles_" after calling the predict method.
    speed_up : str, default='fastest'
        Which speed up technique to use with for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithm for each distance can be obtained by calling the
        `get_speedup_function_names`function.

    Attributes
    ----------
    _X : array, shape (n_cases, n_channels, n_timepoints)
        The input time series stored during the fit method.
    distance_profile_function : function
        The function used to compute the distance profile affected
        during the fit method based on the distance and normalize
        parameters.

    Examples
    --------
    >>> from aeon.similarity_search.query_search import DummyQuerySearch
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = DummyQuerySearch()
    >>> clf.fit(X_train, y_train)
    DummyQuerySearch(...)
    >>> q = X_test[0, :, 5:15]
    >>> y_pred = clf.predict(q)
    """

    def _predict(self, distance_profile, exclusion_size=None):
        """
        Private predict method for DummySimilaritySearch.

        It compute the distance profiles and then returns the best match

        Parameters
        ----------
        distance_profile : array, shape (n_samples, n_timepoints - query_length + 1)
            Precomputed distance profile.
        exclusion_size : int, optional
            This parameter has no effect on this dummy class as we do k=1.

        Returns
        -------
        array
            An array containing the index of the best match between q and _X.

        """
        id_timestamps = np.concatenate(
            [np.arange(distance_profile[i].shape[0]) for i in range(self.n_cases_)]
        )
        id_samples = np.concatenate(
            [[i] * distance_profile[i].shape[0] for i in range(self.n_cases_)]
        )
        # Could use agmin(axis=None) but we need to support unequal length
        distance_profile = np.concatenate(distance_profile)
        id_best = distance_profile.argmin()

        return (id_samples[id_best], id_timestamps[id_best])
