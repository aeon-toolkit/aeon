"""TopKSimilaritySearch."""

__author__ = ["baraline"]

from aeon.similarity_search.base import BaseSimiliaritySearch


class TopKSimilaritySearch(BaseSimiliaritySearch):
    """
    Top-K similarity search method.

    This class implements the Top-K similarity search method, which finds the closest k series 
    to the query series based on a distance function.

    Parameters
    ----------
    k : int, default=1
        The number of nearest matches from Q to return.
    distance : str, default ="euclidean"
        The name of the distance function to use. Options are "euclidean".
    normalize : bool, default = False
        Whether the distance function should be z-normalized.
    store_distance_profile : bool, default = False
        Whether to store the computed distance profile in the attribute
        "_distance_profile" after calling the predict method.

    Attributes
    ----------
    _X : array, shape (n_instances, n_channels, n_timestamps)
        The input time series stored during the fit method.
    distance_profile_function : function
        The function used to compute the distance profile affected
        during the fit method based on the distance and normalize
        parameters.

    Examples
    --------
    >>> from aeon.similarity_search import TopKSimilaritySearch
    >>> import numpy as np
    >>> X = np.random.rand(10, 1, 100)
    >>> model = TopKSimilaritySearch(k=1, distance="euclidean", normalize=True)
    >>> model.fit(X)
    >>> q = np.random.rand(1, 50)
    >>> model.predict(q)
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
        X : array, shape (n_instances, n_channels, n_timestamps)
            Input array to used as database for the similarity search
        y : optional
            Not used.

        Returns
        -------
        self

        """
        return self

    def _predict(self, q, mask):
        """
        Private predict method for TopKSimilaritySearch.
        
        This method computes the distance profiles and returns the top k matches.
        It is called internally by the public predict method.

        It compute the distance profiles and return the top k matches

        Parameters
        ----------
        q :  array, shape (n_channels, q_length)
            Input query used for similarity search.
        mask : array, shape (n_instances, n_channels, n_timestamps - (q_length - 1))
            Boolean mask of the shape of the distance profile indicating for which part
            of it the distance should be computed.

        Returns
        -------
        array
            An array containing the indexes of the best k matches between q and _X.

        """
        if self.normalize:
            distance_profile = self.distance_profile_function(
                self._X,
                q,
                mask,
                self._X_means,
                self._X_stds,
                self._q_means,
                self._q_stds,
            )
        else:
            distance_profile = self.distance_profile_function(self._X, q, mask)

        if self.store_distance_profile:
            self._distance_profile = distance_profile

        # For now, deal with the multidimensional case as "dependent", so we sum.
        distance_profile = distance_profile.sum(axis=1)

        search_size = distance_profile.shape[-1]
        _argsort = distance_profile.argsort(axis=None)[: self.k]

        return [
            (_argsort[i] // search_size, _argsort[i] % search_size)
            for i in range(self.k)
        ]
