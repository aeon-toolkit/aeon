"""Dummy similarity seach estimator."""

__maintainer__ = []
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
    _X : array, shape (n_cases, n_channels, n_timepoints)
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
        super().__init__(
            distance=distance,
            normalize=normalize,
            store_distance_profile=store_distance_profile,
        )

    def _fit(self, X, y):
        """
        Private fit method, does nothing more than the base class.

        Parameters
        ----------
        X : array, shape (n_cases, n_channels, n_timepoints)
            Input array to used as database for the similarity search
        y : optional
            Not used.

        Returns
        -------
        self

        """
        return self

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
        search_size = distance_profile.shape[-1]
        _id_best = distance_profile.argmin(axis=None)
        return [(_id_best // search_size, _id_best % search_size)]
