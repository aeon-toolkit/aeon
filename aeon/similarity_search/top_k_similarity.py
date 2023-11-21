"""TopKSimilaritySearch."""

__author__ = ["baraline"]

from aeon.similarity_search.base import BaseSimiliaritySearch


class TopKSimilaritySearch(BaseSimiliaritySearch):
    """
    Top-K similarity search method.

    Finds the closest k series to the query series based on a distance function.

    Parameters
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
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = TopKSimilaritySearch(k=1)
    >>> clf.fit(X_train, y_train)
    TopKSimilaritySearch(...)
    >>> q = X_test[0, :, 5:15]
    >>> y_pred = clf.predict(q)
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
            Input array to used as database for the similarity search.
        y : optional
            Not used.

        Returns
        -------
        self

        """
        return self

    def _predict(self, distance_profile):
        """
        Private predict method for TopKSimilaritySearch.

        It takes the distance profiles and return the top k matches.

        Parameters
        ----------
        distance_profile : array, shape (n_samples, n_timestamps - q_length + 1)
            Precomputed distance profile.

        Returns
        -------
        array
            An array containing the indexes of the best k matches between q and _X.

        """
        search_size = distance_profile.shape[-1]
        _argsort = distance_profile.argsort(axis=None)[: self.k]

        return [
            (_argsort[i] // search_size, _argsort[i] % search_size)
            for i in range(self.k)
        ]
