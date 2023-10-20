"""TopKSimilaritySearch."""

__author__ = ["baraline"]

import numpy as np

from aeon.similarity_search.base import BaseSimiliaritySearch


class TopKSimilaritySearch(BaseSimiliaritySearch):
    """
    Top-K similarity search method.

    Finds the closest k series to the query series based on a distance function.

    Attributes
    ----------
    k : int, default=1
        The number of nearest matches from Q to return.
    distance : str, default ="euclidean"
        Name of the distance function to use.
    normalize : bool, default = False
        Whether the distance function should be z-normalized.
    store_distance_profile : bool, default = =False.
        Whether to store the computed distance profile in the attribute
        "_distance_profile" after calling the predict method.
    """

    def __init__(
        self, k=1, distance="euclidean", normalize=False, store_distance_profile=False
    ):
        self.k = k
        super(TopKSimilaritySearch, self).__init__(
            distance=distance,
            normalize=normalize,
            store_distance_profile=store_distance_profile,
        )

    def _fit(self, X, y):
        """
        Private fit method, does nothing more than the base class.

        Parameters
        ----------
        X : array, shape (n_cases, n_channels, n_timestamps)
            Input array to used as database for the similarity search
        y : optional
            Not used.

        Returns
        -------
        self

        """
        return self

    def _predict(self, q, q_index=None, exclusion_factor=2):
        """
        Private predict method for TopKSimilaritySearch.

        It compute the distance profiles and return the top k matches

        Parameters
        ----------
        q :  array, shape (n_channels, q_length)
            Input query used for similarity search.
        q_index : tuple, default=None
            Tuple used to specify the index of Q if it is extracted from the input data
            X given during the fit method. Given the tuple (id_sample, id_timestamp),
            the similarity search will define an exclusion zone around the q_index in
            order to avoid matching q with itself. If None, it is considered that the
            query is not extracted from X.
        exclusion_factor : float, default=2.
            The factor to apply to the query length to define the exclusion zone. The
            exclusion zone is define from id_timestamp - q_length//exclusion_factor to
            id_timestamp + q_length//exclusion_factor

        Returns
        -------
        array
            An array containing the indexes of the best k matches between q and _X.

        """
        if self.normalize:
            distance_profile = self.distance_profile_function(
                self._X, q, self._X_means, self._X_stds, self._q_means, self._q_stds
            )
        else:
            distance_profile = self.distance_profile_function(self._X, q)

        if q_index is not None:
            q_length = q.shape[1]
            i_sample, i_timestamp = q_index
            profile_length = distance_profile[i_sample].shape[-1]
            exclusion_LB = max(0, i_timestamp - q_length // exclusion_factor)
            exclusion_UB = min(
                profile_length, i_timestamp + q_length // exclusion_factor
            )
            distance_profile[i_sample][exclusion_LB:exclusion_UB] = np.inf

        if self.store_distance_profile:
            self._distance_profile = distance_profile

        search_size = distance_profile.shape[-1]

        _argsort = distance_profile.argsort(axis=None)[: self.k]

        return [
            (_argsort[i] // search_size, _argsort[i] % search_size)
            for i in range(self.k)
        ]
