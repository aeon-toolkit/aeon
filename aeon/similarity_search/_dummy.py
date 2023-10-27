"""Dummy similarity seach estimator."""

__author__ = ["baraline"]
__all__ = ["DummySimilaritySearch"]


from aeon.similarity_search.base import BaseSimiliaritySearch


class DummySimilaritySearch(BaseSimiliaritySearch):
    """
    DummySimilaritySearch for testing of the BaseSimiliaritySearch class.

    Parameters
    ----------
    distance : str, default ="euclidean"
        Name of the distance function to use.
    normalize : bool, default = False
        Whether the distance function should be z-normalized.
    store_distance_profile : bool, default = =False.
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
    >>> from aeon.similarity_search._dummy import DummySimilaritySearch
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = DummySimilaritySearch()
    >>> clf.fit(X_train, y_train)
    DummySimilaritySearch(...)
    >>> q = X_test[0, :, 5:15]
    >>> y_pred = clf.predict(q)
    """

    def __init__(
        self, distance="euclidean", normalize=False, store_distance_profile=False
    ):
        super(DummySimilaritySearch, self).__init__(
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
        Private predict method for DummySimilaritySearch.

        It compute the distance profiles and then returns the best match

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
            An array containing the index of the best match between q and _X.

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
        search_size = distance_profile.shape[-1]
        distance_profile = distance_profile.sum(axis=1)
        _id_best = distance_profile.argmin(axis=None)

        return [(_id_best // search_size, _id_best % search_size)]
