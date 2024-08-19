"""Naive distance profile computation."""

__maintainer__ = ["baraline"]


from typing import Optional, Union

import numpy as np
from numba.typed import List

from aeon.similarity_search.query_search import QuerySearch


def naive_series_search(
    X: Union[np.ndarray, List],
    S: np.ndarray,
    L: int,
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
    r"""
    Compute a matrix profile in a naive way, by looping through a query search.

    Parameters
    ----------
    X: array shape (n_cases, n_channels, n_timepoints)
        The input samples. If X is an unquel length collection, expect a TypedList
        of 2D arrays of shape (n_channels, n_timepoints)
    S : np.ndarray shape (n_channels, series_length)
        The series used for similarity search. Note that series_length can be equal,
        superior or inferior to n_timepoints, it doesn't matter.
    L : int
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
        An int used to specify the index of S in X, if S is part of X. Otherwise,
        defaults to None, meaning that S is not a sample of X.
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


    Returns
    -------
    Tuple(np.ndarray, 1D array of shape (series_length - L + 1, n_matches), np.ndarray, 2D array of shape (series_length - L + 1, n_matches, 2)) # noqa: E501
        The first array contains the distance between the best matches of the i-th
        subsequence of size L in S and all the subsequences of size L in X.
        The second will contains the sample index and timepoint index of these best
        matches in X.

    """
    search = QuerySearch(
        k=k,
        threshold=threshold,
        distance=distance,
        distance_args=distance_args,
        inverse_distance=inverse_distance,
        normalize=normalize,
        speed_up=speed_up,
        n_jobs=n_jobs,
    )
    search.fit(X)

    results = [
        search.predict(
            S[:, i : i + L],
            X_index=(X_index, i) if X_index is not None else None,
            apply_exclusion_to_result=apply_exclusion_to_result,
            exclusion_factor=exclusion_factor,
        )
        for i in range(S.shape[1] - L + 1)
    ]
    MP = np.empty((S.shape[1] - L + 1, k), dtype=float)
    IP = np.empty((S.shape[1] - L + 1, k, 2), dtype=int)
    for i in range(len(results)):
        MP[i] = results[i][0]
        IP[i] = results[i][1]
    return MP, IP
