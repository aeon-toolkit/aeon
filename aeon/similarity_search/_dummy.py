"""Dummy similarity seach estimator."""

__maintainer__ = []
__all__ = ["DummySimilaritySearch"]


from aeon.similarity_search.base import BaseSeriesSimilaritySearch


class DummySimilaritySearch(BaseSeriesSimilaritySearch):
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
    X_ : array, shape (n_channels, n_timepoints)
        The input time series stored during the fit method.
    distance_profile_function : function
        The function used to compute the distance profile affected
        during the fit method based on the distance and normalize
        parameters.

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
        X : array, shape (n_channels, n_timepoints)
            Not used.
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
        distance_profile : array, shape (n_timepoints - query_length + 1)
            Precomputed distance profile.
        exclusion_size : int, optional
            This parameter has no effect on this dummy class as we do k=1.

        Returns
        -------
        int
            the index of the best match between q and _X.

        """
        return distance_profile.argmin()
