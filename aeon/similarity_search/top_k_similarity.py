"""TopKSimilaritySearch."""

__author__ = ["baraline"]

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

    def _predict(self, q):
        """
        Private predict method for TopKSimilaritySearch.

        It compute the distance profiles and return the top k matches

        Parameters
        ----------
        q :  array, shape (n_channels, q_length)
            Input query used for similarity search.

        Raises
        ------
        TypeError
            If the input q array is not 2D raise an error.

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
        # Would creating base distance profile classes be relevant to force the same
        # interface for normalized / non normalized distance profiles ?
        if self.store_distance_profile:
            self._distance_profile = distance_profile

        search_size = distance_profile.shape[-1]

        _argsort = distance_profile.argsort(axis=None)[: self.k]

        # return is [(id_sample, id_timestamp)]
        #    -> candidate is X[id_sample, :, id_timestamps:id_timestamps+q_length]
        return [
            (_argsort[i] // search_size, _argsort[i] % search_size)
            for i in range(self.k)
        ]
