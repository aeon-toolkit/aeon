"""Implementation of STAMP for euclidean and squared euclidean distance profile."""

__maintainer__ = ["baraline"]


from typing import Optional, Union

import numpy as np
from numba import njit, prange
from numba.typed import List

# def stamp_euclidean_matrix_profile()


def stamp_squared_matrix_profile(
    X: Union[np.ndarray, List],
    T: np.ndarray,
    length: int,
    k: int = 1,
    threshold: float = np.inf,
    distance: str = "euclidean",
    distance_args: Optional[dict] = None,
    inverse_distance: bool = False,
    normalize: bool = False,
    speed_up: str = "fastest",
    n_jobs: int = 1,
    X_index: Optional[int] = None,
    exclusion_factor: float = 2.0,
    apply_exclusion_to_result: bool = True,
):
    """
    Compute a squared euclidean matrix profile using STAMP [1]_.

    This improves on the naive matrix profile by updating the dot products for each
    sucessive query in T instead of recomputing them.

    Parameters
    ----------
    X:  np.ndarray, 3D array of shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    T : np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    length : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    k : int, default=1
        The number of best matches to return during predict for each subsequence.
    threshold : float, default=np.inf
        The number of best matches to return during predict for each subsequence.
    distance : str, default="euclidean"
        Name of the distance function to use. A list of valid strings can be found in
        the documentation for :func:`aeon.distances.get_distance_function`.
        If a callable is passed it must either be a python function or numba function
        with nopython=True, that takes two 1d numpy arrays as input and returns a float.
    distance_args : dict, default=None
        Optional keyword arguments for the distance function.
    normalize : bool, default=False
        Whether the distance function should be z-normalized.
    speed_up : str, default='fastest'
        Which speed up technique to use with for the selected distance
        function. By default, the fastest algorithm is used. A list of available
        algorithm for each distance can be obtained by calling the
        `get_speedup_function_names` function.
    inverse_distance : bool, default=False
        If True, the matching will be made on the inverse of the distance, and thus, the
        worst matches to the query will be returned instead of the best ones.
    n_jobs : int, default=1
        Number of parallel jobs to use.
    X_index : int, default=None
        An int used to specify the index of T in X, if T is part of X. Otherwise,
        defaults to None, meaning that T is not a sample of X.
    exclusion_factor : float, default=2.
        The factor to apply to the query length to define the exclusion zone. The
        exclusion zone is define from
        :math:`id_timestamp - query_length//exclusion_factor` to
        :math:`id_timestamp + query_length//exclusion_factor`. This also applies to
        the matching conditions defined by child classes. For example, with
        TopKSimilaritySearch, the k best matches are also subject to the exclusion
        zone, but with :math:`id_timestamp` the index of one of the k matches.
    apply_exclusion_to_result: bool, default=True
        Wheter to apply the exclusion factor to the output of the similarity search.
        This means that two matches of the query from the same sample must be at
        least spaced by +/- :math:`query_length//exclusion_factor`.
        This can avoid pathological matching where, for example if we extract the
        best two matches, there is a high chance that if the best match is located
        at :math:`id_timestamp`, the second best match will be located at
        :math:`id_timestamp` +/- 1, as they both share all their values except one.

    References
    ----------
    .. [1] Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View
    that Includes Motifs, Discords and Shapelets. Chin-Chia Michael Yeh, Yan Zhu,
    Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva,
    Abdullah Mueen, Eamonn Keogh (2016). IEEE ICDM 2016

    Returns
    -------
    Tuple(ndarray, ndarray)
        The first array, of shape ``(series_length - length + 1, n_matches)``,
        contains the distance between all the queries of size length and their best
        matches in X_. The second array, of shape
        ``(series_length - L + 1, n_matches, 2)``, contains the indexes of these
        matches as ``(id_sample, id_timepoint)``. The corresponding match can be
        retrieved as ``X_[id_sample, :, id_timepoint : id_timepoint + length]``.

    """
    pass


@njit(cache=True, fastmath=True)
def _update_dot_product(
    X,
    T,
    XT_products,
    L,
    i_query,
):
    """
    Update dot products of the i-th query of size L in T from the dot products of i-1.

    Parameters
    ----------
    X: np.ndarray, 2D array of shape (n_channels, n_timepoints)
        Input time series on which the sliding dot product is computed.
    T: np.ndarray, 2D array of shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
        The length of the subsequences considered during the search. This parameter
        cannot be larger than n_timepoints and series_length.
    i_query : int
        Query starting index in T.

    Returns
    -------
    XT_products : np.ndarray, 2D array of shape (n_channels, n_timepoints - L + 1)
        Sliding dot product between the i-th subsequence of size L in T and X..

    """
    n_channels = T.shape[0]
    n_timepoints = X.shape[1]
    n_candidates = n_timepoints - L + 1

    if i_query > 0:
        Q = T[:, i_query : i_query + L]
        # first element of all 0 to n-1 candidates * first element of previous query
        _a1 = X[:, : n_candidates - 1] * T[:, i_query - 1][:, np.newaxis]
        # last element of all 1 to n candidates * last element of current query
        _a2 = X[:, L : L - 1 + n_candidates] * T[:, i_query + L - 1][:, np.newaxis]

        XT_products[:, 1:] = XT_products[:, :-1] - _a1 + _a2
        # Compute first dot product
        for i_ft in prange(n_channels):
            XT_products[i_ft, 0] = np.sum(Q[i_ft] * X[i_ft, :L])
    return XT_products
