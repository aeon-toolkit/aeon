"""TopKSimilaritySearch."""

__maintainer__ = []

import warnings

import numpy as np

from aeon.similarity_search.base import BaseSimiliaritySearch


class TopKSimilaritySearch(BaseSimiliaritySearch):
    """
    Top-K similarity search method.

    Finds the closest k series to the query series based on a distance function.

    Parameters
    ----------
    k : int, default=1
        The number of nearest matches from Q to return.
     distance : str, default="euclidean"
         Name of the distance function to use. A list of valid strings can be found in
         the documentation for :func:`aeon.distances.get_distance_function`.
         If a callable is passed it must either be a python function or numba function
         with nopython=True, that takes two 1d numpy arrays as input and returns a
         float.
     distance_args : dict, default=None
         Optional keyword arguments for the distance function.
     normalize : bool, default=False
         Whether the distance function should be z-normalized.
     store_distance_profile : bool, default=False.
         Whether to store the computed distance profile in the attribute
         "_distance_profile" after calling the predict method.
     speed_up : str, default=None
         Which speed up technique to use with for the selected distance function.

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
    >>> from aeon.similarity_search import TopKSimilaritySearch
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = TopKSimilaritySearch(k=1)
    >>> clf.fit(X_train, y_train)
    TopKSimilaritySearch(...)
    >>> q = X_test[0, :, 5:15]
    >>> y_pred = clf.predict(q)

    Notes
    -----
    For now, the multivariate case is only treated as independent.
    Distances are computed for each channel independently and then
    summed together.
    """

    def __init__(
        self,
        k=1,
        distance="euclidean",
        distance_args=None,
        normalize=False,
        speed_up=None,
        store_distance_profile=False,
    ):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                f"Got k={k} for TopKSimilaritySearch. Parameter k can only be an"
                "integer superior or equal to 1"
            )
        self.k = k
        super().__init__(
            distance=distance,
            distance_args=distance_args,
            normalize=normalize,
            speed_up=speed_up,
            store_distance_profile=store_distance_profile,
        )

    def _fit(self, X, y):
        """
        Private fit method, does nothing more than the base class.

        Parameters
        ----------
        X : array, shape (n_cases, n_channels, n_timepoints)
            Input array to used as database for the similarity search.
        y : optional
            Not used.

        Returns
        -------
        self

        """
        return self

    def _predict(self, distance_profile, exclusion_size=None):
        """
        Private predict method for TopKSimilaritySearch.

        It takes the distance profiles and return the top k matches.

        Parameters
        ----------
        distance_profile : array, shape (n_cases, n_timepoints - query_length + 1)
            Precomputed distance profile.
        exclusion_size : int, optional
            The size of the exclusion zone used to prevent returning as top k candidates
            the ones that are close to each other (for example i and i+1).
            It is used to define a region between
            :math:`id_timestamp - exclusion_size` and
            :math:`id_timestamp + exclusion_size` which cannot be returned
            as best match if :math:`id_timestamp` was already selected. By default,
            the value None means that this is not used.

        Returns
        -------
        array
            An array containing the indexes of the best k matches between q and _X.

        """
        search_size = distance_profile.shape[-1]
        _argsort = distance_profile.argsort(axis=None)
        _argsort = np.asarray(
            [
                [_argsort[i] // search_size, _argsort[i] % search_size]
                for i in range(len(_argsort))
            ],
            dtype=int,
        )
        if _argsort.shape[0] < self.k:
            _k = _argsort.shape[0]
            warnings.warn(
                f"The number of possible match is {_argsort.shape[0]}, but got"
                f"k={self.k}. The number of returned match will be {_argsort.shape[0]}",
                stacklevel=2,
            )
        else:
            _k = self.k

        if exclusion_size is None:
            return _argsort[:_k]
        else:
            top_k = np.zeros((_k, 2), dtype=int) - 1
            top_k[0] = _argsort[0, :]

            n_inserted = 1
            i_current = 1

            while n_inserted < _k and i_current < _argsort.shape[0]:
                candidate_sample, candidate_timestamp = _argsort[i_current]

                insert = True
                is_from_same_sample = top_k[:, 0] == candidate_sample
                if np.any(is_from_same_sample):
                    LB = candidate_timestamp >= (
                        top_k[is_from_same_sample, 1] - exclusion_size
                    )
                    UB = candidate_timestamp <= (
                        top_k[is_from_same_sample, 1] + exclusion_size
                    )
                    if np.any(UB & LB):
                        insert = False

                if insert:
                    top_k[n_inserted] = _argsort[i_current]
                    n_inserted += 1
                i_current += 1

            return top_k[:n_inserted]
