"""K-Shape style shape extraction and averaging.

This module implements a K-Shape-like shape extraction procedure that
reuses the SBD utilities from :mod:`aeon.distances._sbd`.

The core idea (Paparrizos & Gravano, SIGMOD 2015) is:

1. Align each series in a cluster to the current centre using SBD
   (max normalized cross-correlation over all circular shifts).
2. Z-score the aligned series.
3. Compute a covariance-like matrix over time indices and extract the
   principal eigenvector of a centred version of this matrix as the new
   "shape" (centroid).
4. Fix the sign of the eigenvector to minimise distance to the aligned
   series, and z-score the final centroid.

For multivariate series, the shape extraction is performed independently
per channel.

This implementation is for fixed clusters: given a collection of time
series assigned to one cluster, :func:`shape_extraction_average` returns
the K-Shape-style shape centre of that cluster.
"""

__maintainer__ = []

from typing import Literal

import numpy as np
from numpy.linalg import eigh
from numba import njit
from sklearn.utils import check_random_state

from aeon.distances._sbd import (
    _univariate_sbd_align_to_center,
    _zscore_1d,
    sbd_pairwise_distance,
)
from aeon.utils.numba._threading import threaded

__all__ = ["shape_extraction_average"]


@njit(cache=True, fastmath=True)
def _kshape_covariance_univariate_numba(
    X: np.ndarray,
    centre: np.ndarray,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated part of K-Shape shape extraction for one channel.

    1. Align each series to the current centre using SBD (max NCC).
    2. Z-score each aligned series.
    3. Compute S = aligned.T @ aligned.
    """
    n_cases, n_timepoints = X.shape
    aligned = np.empty_like(X)

    for i in range(n_cases):
        aligned[i] = _univariate_sbd_align_to_center(centre, X[i], standardize)
        aligned[i] = _zscore_1d(aligned[i])

    S = aligned.T @ aligned
    return S, aligned


@njit(cache=True, fastmath=True)
def _kshape_extract_shape_univariate(
    X: np.ndarray,
    centre: np.ndarray,
    standardize: bool = True,
) -> np.ndarray:
    """K-Shape shape extraction for a single univariate channel (numba-friendly).

    Parameters
    ----------
    X : (n_cases, n_timepoints)
    centre : (n_timepoints,)
    standardize : bool

    Returns
    -------
    shape : (n_timepoints,)
    """
    n_cases, n_timepoints = X.shape

    if n_cases == 0:
        return centre.copy()

    # 1–3) align, z-score, and covariance-like matrix
    S, aligned = _kshape_covariance_univariate_numba(X, centre, standardize)

    # 4) construct centring matrix P and M = P S P
    columns = n_timepoints
    P = np.zeros((columns, columns))
    inv_cols = 1.0 / columns

    # P = I - (1/columns) * 1 1^T
    for i in range(columns):
        for j in range(columns):
            if i == j:
                P[i, j] = 1.0 - inv_cols
            else:
                P[i, j] = -inv_cols

    # M = P S P
    # First T = S P
    T = np.zeros_like(S)
    for i in range(columns):
        for j in range(columns):
            acc = 0.0
            for k in range(columns):
                acc += S[i, k] * P[k, j]
            T[i, j] = acc

    M = np.zeros_like(S)
    for i in range(columns):
        for j in range(columns):
            acc = 0.0
            for k in range(columns):
                acc += P[i, k] * T[k, j]
            M[i, j] = acc

    # 5) principal eigenvector of M
    vals, vecs = eigh(M)
    shape = vecs[:, -1]  # last eigenvector

    # 6) choose sign to minimise distance to aligned series
    #    compute sum_i ||aligned[i] - shape|| and ||aligned[i] + shape||
    dist1 = 0.0
    dist2 = 0.0
    for i in range(n_cases):
        # norm of aligned[i] - shape
        acc1 = 0.0
        acc2 = 0.0
        for t in range(n_timepoints):
            d1 = aligned[i, t] - shape[t]
            d2 = aligned[i, t] + shape[t]
            acc1 += d1 * d1
            acc2 += d2 * d2
        dist1 += np.sqrt(acc1)
        dist2 += np.sqrt(acc2)

    if dist1 >= dist2:
        for t in range(n_timepoints):
            shape[t] = -shape[t]

    # 7) final z-score of shape
    shape = _zscore_1d(shape)
    return shape


def _kshape_one_iter(
    X: np.ndarray,
    centre: np.ndarray,
    standardize: bool = True,
) -> np.ndarray:
    """One K-Shape-style update of the shape centre.

    Parameters
    ----------
    X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
        Cluster time series.
    centre : np.ndarray, shape (n_channels, n_timepoints)
        Current shape centre.
    standardize : bool, default=True
        Standardization flag passed to the SBD alignment step.

    Returns
    -------
    new_centre : np.ndarray, shape (n_channels, n_timepoints)
        Updated shape centre after one K-Shape iteration.
    """
    X = np.asarray(X, dtype=float)
    centre = np.asarray(centre, dtype=float)

    n_cases, n_channels, n_timepoints = X.shape
    new_centre = np.empty_like(centre)

    for c in range(n_channels):
        # Extract all series for this channel: (N, T)
        X_c = X[:, c, :]
        centre_c = centre[c]
        new_centre[c] = _kshape_extract_shape_univariate(
            X_c,
            centre_c,
            standardize=standardize,
        )

    return new_centre


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


@threaded
def shape_extraction_average(
    X: np.ndarray,
    max_iters: int = 10,
    tol: float = 1e-5,
    init_centre: Literal["mean", "random"] | np.ndarray = "random",
    standardize: bool = True,
    random_state: int | None = None,
    verbose: bool = False,
    n_jobs: int = 1,
    return_distances_to_centre: bool = False,
    return_cost: bool = False,
):
    """Compute a K-Shape style shape-extraction average of time series.

    This function assumes all series in ``X`` belong to a single cluster and
    returns the K-Shape-style shape centre of that cluster. It iteratively:

    1. Aligns each series to the current centre using SBD (max NCC).
    2. Performs shape extraction per channel via the principal eigenvector
       of a centred covariance-like matrix.
    3. Optionally computes the SBD cost to check for convergence.

    Parameters
    ----------
    X : np.ndarray, shape (n_cases, n_channels, n_timepoints) or (n_cases, n_timepoints)
        Collection of time series to average. If 2D, it is internally reshaped to
        (n_cases, 1, n_timepoints).
    max_iters : int, default=10
        Maximum number of K-Shape update iterations.
        Setting ``max_iters=1`` corresponds to a single shape extraction step.
    tol : float, default=1e-5
        Early-stopping tolerance on the SBD cost between iterations. If the
        absolute change in cost is smaller than ``tol``, the procedure stops.
    init_centre : {"mean", "random"} or np.ndarray, default="random"
        Initial shape centre. If a string is provided, it specifies the
        initialisation strategy:

        - "mean": arithmetic mean over series in X.
        - "random": a single random series from X.

        If a 2D array is provided, it is used directly as the starting
        centre and must have shape (n_channels, n_timepoints).
    standardize : bool, default=True
        Whether to standardize in the SBD alignment step. This should match
        the ``standardize`` argument used with SBD elsewhere for consistency.
    random_state : int or None, default=None
        Random seed used when ``init_centre="random"``.
    verbose : bool, default=False
        If True, prints progress information.
    n_jobs : int, default=1
        Number of jobs used internally by :func:`sbd_pairwise_distance` when
        computing the cost (distances to the centre).
    return_distances_to_centre : bool, default=False
        If True, also return the distances between each time series and the
        shape centre (under SBD).
    return_cost : bool, default=False
        If True, also return the total cost (sum of SBD distances).

    Returns
    -------
    centre : np.ndarray, shape (n_channels, n_timepoints)
        The K-Shape style shape centre of the input time series.
    distances_to_centre : np.ndarray, shape (n_cases,), optional
        Returned if ``return_distances_to_centre=True``.
    cost : float, optional
        Returned if ``return_cost=True``.

    References
    ----------
    Paparrizos, J., & Gravano, L. (2015).
    "k-Shape: Efficient and Accurate Clustering of Time Series."
    Proceedings of the 2015 ACM SIGMOD.
    """
    X = np.asarray(X, dtype=float)

    # Handle trivial cases
    if X.shape[0] <= 1:
        centre = X[0] if X.ndim == 3 else X
        if X.ndim == 2:
            centre = centre[None, :]  # (T,) -> (1, T)
        if return_distances_to_centre and return_cost:
            return centre, np.zeros(X.shape[0]), 0.0
        elif return_distances_to_centre:
            return centre, np.zeros(X.shape[0])
        elif return_cost:
            return centre, 0.0
        return centre

    # Ensure shape (N, C, T)
    if X.ndim == 2:
        # (N, T) -> (N, 1, T)
        X = X[:, None, :]
    elif X.ndim != 3:
        raise ValueError(
            f"X must have shape (N, T) or (N, C, T), got {X.shape} (ndim={X.ndim})"
        )

    n_cases, n_channels, n_timepoints = X.shape
    rng = check_random_state(random_state)

    # Initialise centre
    if isinstance(init_centre, np.ndarray):
        centre = np.asarray(init_centre, dtype=float)
        if centre.ndim == 1:
            centre = centre[None, :]  # (T,) -> (1, T)
        if centre.shape != (n_channels, n_timepoints):
            raise ValueError(
                f"init_centre has shape {centre.shape}, "
                f"expected {(n_channels, n_timepoints)}"
            )
    else:
        if init_centre == "mean":
            centre = X.mean(axis=0)
        elif init_centre == "random":
            idx0 = rng.randint(n_cases)
            centre = X[idx0].copy()
        else:
            raise ValueError(
                f"Unknown init_centre={init_centre!r}. "
                'Use "mean", "random" or an array.'
            )

    prev_cost = float("inf")
    prev_centre = centre.copy()
    distances_to_centre = None
    cost = prev_cost

    for i in range(max_iters):
        # One K-Shape update step (internally calls numba-accelerated parts)
        centre = _kshape_one_iter(
            X,
            centre=centre,
            standardize=standardize,
        )

        # Compute SBD distances to current centre for cost
        # centre[None, ...] has shape (1, C, T)
        dists = sbd_pairwise_distance(
            X,
            centre[None, ...],
            standardize=standardize,
            n_jobs=n_jobs,
        ).reshape(n_cases)
        cost = float(np.sum(dists))

        if verbose:
            print(f"[KShape-ShapeExtraction] epoch {i}, cost {cost}")  # noqa: T201

        # Early stopping: small change or cost increase
        if abs(prev_cost - cost) < tol:
            if prev_cost < cost:
                centre = prev_centre
                dists = sbd_pairwise_distance(
                    X,
                    centre[None, ...],
                    standardize=standardize,
                    n_jobs=n_jobs,
                ).reshape(n_cases)
                cost = float(np.sum(dists))
            if verbose:
                print(
                    f"[KShape-ShapeExtraction] epoch {i}, early convergence: "
                    f"{cost} - {prev_cost} < tol={tol}"
                )
            distances_to_centre = dists
            break
        elif prev_cost < cost:
            centre = prev_centre
            dists = sbd_pairwise_distance(
                X,
                centre[None, ...],
                standardize=standardize,
                n_jobs=n_jobs,
            ).reshape(n_cases)
            cost = float(np.sum(dists))
            if verbose:
                print(
                    f"[KShape-ShapeExtraction] epoch {i}, cost increased: "
                    f"{cost} > previous cost {prev_cost}"
                )
            distances_to_centre = dists
            break
        else:
            prev_cost = cost
            prev_centre = centre.copy()
            distances_to_centre = dists

    if verbose:
        print(
            f"[KShape-ShapeExtraction] converged at epoch {i}, cost {cost}"
        )  # noqa: T201

    if return_distances_to_centre and return_cost:
        return centre, distances_to_centre, cost
    elif return_distances_to_centre:
        return centre, distances_to_centre
    elif return_cost:
        return centre, cost
    return centre