r"""Sequence Weighted Alignment (Swale) distance between two time series."""

__maintainer__ = []

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.pointwise._euclidean import _univariate_euclidean_distance
from aeon.utils.conversion._convert_collection import _convert_collection_to_numba_list
from aeon.utils.decorators.numba_threading import numba_thread_handler
from aeon.utils.validation.collection import _is_numpy_list_multivariate


@njit(cache=True, fastmath=True)
def swale_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    epsilon: float = 1.0,
    match_reward: float = 50.0,
    gap_penalty: float = -8.0,
    itakura_max_slope: float | None = None,
) -> float:
    r"""Return the Swale distance between x and y.

    The Sequence Weighted Alignment model (Swale) [1]_ is an
    :math:`\epsilon`-threshold based similarity measure that, unlike LCSS, combines a
    reward for matching points with a penalty for gaps. Two points are considered a
    match if their distance is within ``epsilon`` in every channel; a match contributes
    ``match_reward`` to the score, while a gap in either series contributes
    ``gap_penalty``. The score is defined by the recurrence

    .. math::
        Swale(R, S) = \begin{cases}
            n \cdot g & \text{if } m = 0 \\
            m \cdot g & \text{if } n = 0 \\
            r + Swale(Rest(R), Rest(S)) & \text{if } |r_1 - s_1| \le \epsilon \\
            \max\{g + Swale(Rest(R), S),\ g + Swale(R, Rest(S))\} & \text{otherwise}
        \end{cases}

    where :math:`r` is ``match_reward`` and :math:`g` is ``gap_penalty``. This score is
    a similarity: it is maximised (``match_reward * min(m, n)``) when the shorter series
    matches the longer one entirely. To return a distance, the score is normalised in
    the same way as LCSS, ``1 - score / (match_reward * min(m, n))``, so that identical
    series have distance ``0`` and larger values indicate greater dissimilarity. This
    requires ``match_reward > 0``.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold: two points match if their distance is at most ``epsilon``.
    match_reward : float, default=50.
        The reward added to the score for each matching pair of points.
    gap_penalty : float, default=-8.
        The penalty added to the score for each gap. Should be negative to penalise
        gaps. The defaults follow the reward/penalty ratio used in [1]_.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    float
        The Swale distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    References
    ----------
    .. [1] Michael D. Morse and Jignesh M. Patel. 2007. An efficient and accurate
        method for evaluating time series similarity. In Proceedings of the 2007 ACM
        SIGMOD International Conference on Management of Data (SIGMOD '07), 569-580.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import swale_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> swale_distance(x, x)
    0.0
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _swale_distance(
            _x, _y, bounding_matrix, epsilon, match_reward, gap_penalty
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _swale_distance(
            x, y, bounding_matrix, epsilon, match_reward, gap_penalty
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def swale_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    epsilon: float = 1.0,
    match_reward: float = 50.0,
    gap_penalty: float = -8.0,
    itakura_max_slope: float | None = None,
) -> np.ndarray:
    r"""Return the Swale cost matrix between x and y.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold: two points match if their distance is at most ``epsilon``.
    match_reward : float, default=50.
        The reward added to the score for each matching pair of points.
    gap_penalty : float, default=-8.
        The penalty added to the score for each gap. Should be negative to penalise
        gaps.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    np.ndarray
        The Swale cost matrix between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import swale_cost_matrix
    >>> x = np.array([[1, 2, 3]])
    >>> swale_cost_matrix(x, x)
    array([[  0.,  -8., -16., -24.],
           [ -8.,  50.,  42.,  34.],
           [-16.,  42., 100.,  92.],
           [-24.,  34.,  92., 150.]])
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _swale_cost_matrix(
            _x, _y, bounding_matrix, epsilon, match_reward, gap_penalty
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _swale_cost_matrix(
            x, y, bounding_matrix, epsilon, match_reward, gap_penalty
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _swale_distance(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: float,
    match_reward: float,
    gap_penalty: float,
) -> float:
    cost_matrix = _swale_cost_matrix(
        x, y, bounding_matrix, epsilon, match_reward, gap_penalty
    )
    score = cost_matrix[x.shape[1], y.shape[1]]
    # Convert the Swale similarity score into a distance, mirroring LCSS: the maximum
    # attainable score is ``match_reward * min(m, n)`` (every point of the shorter
    # series matched), so ``1 - score / (match_reward * min(m, n))`` is 0 for identical
    # series and non-negative in general (the score can never exceed that maximum).
    max_score = match_reward * min(x.shape[1], y.shape[1])
    return 1.0 - (score / max_score)


@njit(cache=True, fastmath=True)
def _swale_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: float,
    match_reward: float,
    gap_penalty: float,
) -> np.ndarray:
    x_size = x.shape[1]
    y_size = y.shape[1]

    cost_matrix = np.full((x_size + 1, y_size + 1), -np.inf)
    cost_matrix[0, 0] = 0.0
    for i in range(1, x_size + 1):
        cost_matrix[i, 0] = i * gap_penalty
    for j in range(1, y_size + 1):
        cost_matrix[0, j] = j * gap_penalty

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if bounding_matrix[i - 1, j - 1]:
                if _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
                    cost_matrix[i, j] = cost_matrix[i - 1, j - 1] + match_reward
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i - 1, j] + gap_penalty,
                        cost_matrix[i, j - 1] + gap_penalty,
                    )
    return cost_matrix


@numba_thread_handler
def swale_pairwise_distance(
    X: np.ndarray | list[np.ndarray],
    y: np.ndarray | list[np.ndarray] | None = None,
    window: float | None = None,
    epsilon: float = 1.0,
    match_reward: float = 50.0,
    gap_penalty: float = -8.0,
    itakura_max_slope: float | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute the Swale pairwise distance between a set of time series.

    Parameters
    ----------
    X : np.ndarray or List of np.ndarray
        A collection of time series instances  of shape ``(n_cases, n_timepoints)``
        or ``(n_cases, n_channels, n_timepoints)``.
    y : np.ndarray or List of np.ndarray or None, default=None
        A single series or a collection of time series of shape ``(m_timepoints,)`` or
        ``(m_cases, m_timepoints)`` or ``(m_cases, m_channels, m_timepoints)``.
        If None, then the swale pairwise distance between the instances of X is
        calculated.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold: two points match if their distance is at most ``epsilon``.
    match_reward : float, default=50.
        The reward added to the score for each matching pair of points.
    gap_penalty : float, default=-8.
        The penalty added to the score for each gap. Should be negative to penalise
        gaps.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    n_jobs : int, default=1
        The number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. If 1, then the function is executed in a single
        thread. If greater than 1, then the function is executed in parallel.

    Returns
    -------
    np.ndarray (n_cases, n_cases)
        Swale pairwise matrix between the instances of X.

    Raises
    ------
    ValueError
        If X is not 2D or 3D array when only passing X.
        If X and y are not 1D, 2D or 3D arrays when passing both X and y.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import swale_pairwise_distance
    >>> X = np.array([[[1, 2, 3]],[[4, 5, 6]], [[7, 8, 9]]])
    >>> swale_pairwise_distance(X)
    array([[0.  , 0.88, 1.32],
           [0.88, 0.  , 0.88],
           [1.32, 0.88, 0.  ]])
    """
    multivariate_conversion = _is_numpy_list_multivariate(X, y)
    _X, unequal_length = _convert_collection_to_numba_list(
        X, "X", multivariate_conversion
    )
    if y is None:
        return _swale_pairwise_distance(
            _X,
            window,
            epsilon,
            match_reward,
            gap_penalty,
            itakura_max_slope,
            unequal_length,
        )
    _y, unequal_length = _convert_collection_to_numba_list(
        y, "y", multivariate_conversion
    )
    return _swale_from_multiple_to_multiple_distance(
        _X,
        _y,
        window,
        epsilon,
        match_reward,
        gap_penalty,
        itakura_max_slope,
        unequal_length,
    )


@njit(cache=True, fastmath=True, parallel=True)
def _swale_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: float | None,
    epsilon: float,
    match_reward: float,
    gap_penalty: float,
    itakura_max_slope: float | None,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))
    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(i, n_cases):
            x1, x2 = X[i], X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _swale_distance(
                x1, x2, bounding_matrix, epsilon, match_reward, gap_penalty
            )
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=True, fastmath=True, parallel=True)
def _swale_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: float | None,
    epsilon: float,
    match_reward: float,
    gap_penalty: float,
    itakura_max_slope: float | None,
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )
    for i in prange(n_cases):
        for j in range(m_cases):
            x1, y1 = x[i], y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _swale_distance(
                x1, y1, bounding_matrix, epsilon, match_reward, gap_penalty
            )
    return distances


@njit(cache=True, fastmath=True)
def swale_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: float | None = None,
    epsilon: float = 1.0,
    match_reward: float = 50.0,
    gap_penalty: float = -8.0,
    itakura_max_slope: float | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Compute the Swale alignment path between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, shape ``(n_channels, n_timepoints)`` or ``(n_timepoints,)``.
    y : np.ndarray
        Second time series, shape ``(m_channels, m_timepoints)`` or ``(m_timepoints,)``.
    window : float, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used.
    epsilon : float, default=1.
        Matching threshold: two points match if their distance is at most ``epsilon``.
    match_reward : float, default=50.
        The reward added to the score for each matching pair of points.
    gap_penalty : float, default=-8.
        The penalty added to the score for each gap. Should be negative to penalise
        gaps.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.

    Returns
    -------
    List[Tuple[int, int]]
        The alignment path between the two time series where each element is a tuple
        of the index in x and the index in y that have the best alignment according
        to the cost matrix.
    float
        The Swale distance between the two time series.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.distances import swale_alignment_path
    >>> x = np.array([[1, 2, 3, 4]])
    >>> y = np.array([[1, 2, 3, 4]])
    >>> path, score = swale_alignment_path(x, y)
    >>> path
    [(0, 0), (1, 1), (2, 2), (3, 3)]
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
    elif x.ndim == 2 and y.ndim == 2:
        _x = x
        _y = y
    else:
        raise ValueError("x and y must be 1D or 2D arrays")

    bounding_matrix = create_bounding_matrix(
        _x.shape[1], _y.shape[1], window, itakura_max_slope
    )
    cost_matrix = _swale_cost_matrix(
        _x, _y, bounding_matrix, epsilon, match_reward, gap_penalty
    )
    path = _swale_return_path(_x, _y, cost_matrix, epsilon, match_reward, gap_penalty)
    score = cost_matrix[_x.shape[1], _y.shape[1]]
    max_score = match_reward * min(_x.shape[1], _y.shape[1])
    return path, 1.0 - (score / max_score)


@njit(cache=True, fastmath=True)
def _swale_return_path(
    x: np.ndarray,
    y: np.ndarray,
    cost_matrix: np.ndarray,
    epsilon: float,
    match_reward: float,
    gap_penalty: float,
) -> list[tuple[int, int]]:
    i = x.shape[1]
    j = y.shape[1]
    path = []
    while i > 0 and j > 0:
        if _univariate_euclidean_distance(x[:, i - 1], y[:, j - 1]) <= epsilon:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif cost_matrix[i - 1, j] >= cost_matrix[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return path[::-1]
