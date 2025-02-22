r"""Dynamic time warping with Global Invariances (DTW-GI) between two time series."""

__maintainer__ = []

from typing import Optional

import numpy as np
import scipy.linalg

from aeon.distances.elastic._dtw import dtw_alignment_path


def _path2mat(path, x_ntimepoints, y_ntimepoints):
    r"""Convert a warping alignment path to a binary warping matrix."""
    w = np.zeros((x_ntimepoints, y_ntimepoints))
    for i, j in path:
        w[i, j] = 1
    return w


def dtw_gi(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
    init_p=None,
    max_iter=20,
    use_bias=False,
):
    r"""
    Compute Dynamic Time Warping with Global Invariance between the two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second time series, either univariate, shape ``(n_timepoints,)``, or
        multivariate, shape ``(n_channels, n_timepoints)``.
    window : float or None, default=None
        The window to use for the bounding matrix. If None, no bounding matrix
        is used. window is a percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    itakura_max_slope : float, default=None
        Maximum slope as a proportion of the number of time points used to create
        Itakura parallelogram on the bounding matrix. Must be between 0. and 1.
    init_p : array-like of shape (x_nchannels, y_nchannels), default=None
        Initial linear transformation. If None, the identity matrix is used.
    max_iter : int, default=20
        Maximum number of iterations for the iterative optimization.
    use_bias : bool, default=False
        If True, the feature space map is affine (with a bias term).

    Returns
    -------
    - w_pi: binary warping matrix of shape (n0, n1)
    - p: the final linear (Stiefel) matrix of shape (x_nchannels, y_nchannels)
    - cost: final DTW cost considering global invariances

    If use_bias is True, also returns:
      - bias

    """
    if x.ndim == 1:
        _x = x.reshape((1, -1))
        if y.ndim == 1:
            _y = y.reshape((1, -1))
        return dtw_gi(_x, _y, window, itakura_max_slope, init_p, max_iter, use_bias)

    if y.ndim == 1:
        _y = y.reshape((1, -1))
        if x.ndim == 1:
            _x = x.reshape((1, -1))
        return dtw_gi(x, _y, window, itakura_max_slope, init_p, max_iter, use_bias)

    x_ = x
    y_ = y

    x_nchannels, x_ntimepoints = x_.shape
    y_nchannels, y_ntimepoints = y_.shape

    x_m = x_.mean(axis=1, keepdims=True)
    y_m = y_.mean(axis=1, keepdims=True)

    w_pi = np.zeros((x_ntimepoints, y_ntimepoints))
    if init_p is None:
        p = np.eye(x_nchannels, y_nchannels)
    else:
        p = init_p
    bias = np.zeros((x_nchannels, 1))

    for _ in range(1, max_iter + 1):
        w_pi_old = w_pi.copy()
        y_transformed = p.dot(y_) + bias

        path, cost = dtw_alignment_path(x_, y_transformed)
        w_pi = _path2mat(path, x_ntimepoints, y_ntimepoints)

        if np.allclose(w_pi, w_pi_old):
            break

        if use_bias:
            m = (x_ - x_m).dot(w_pi).dot((y_ - y_m).T)
        else:
            m = x_.dot(w_pi).dot(y_.T)

        u, sigma, vt = scipy.linalg.svd(m, full_matrices=False)
        p = u.dot(vt)
        if use_bias:
            bias = x_m - p.dot(y_m)

    y_trans = p.dot(y_) + bias
    path, cost = dtw_alignment_path(x_, y_trans)

    if use_bias:
        return w_pi, p, bias, cost
    else:
        return w_pi, p, cost
