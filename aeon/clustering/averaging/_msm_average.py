"""Compute the MSM exact barycenter average of time series."""

__maintainer__ = []

import warnings

import numpy as np
from numba import njit

from aeon.distances import msm_distance


def msm_barycenter_average(
    X: np.ndarray,
    c: float = 1.0,
    window: int | None = None,
    init_barycenter: np.ndarray | str = "mean",
    weights: np.ndarray | None = None,
    precomputed_medoids_pairwise_distance: np.ndarray | None = None,
    verbose: bool = False,
    random_state: int | None = None,
    n_jobs: int = 1,
    return_distances_to_center: bool = False,
    return_cost: bool = False,
    **kwargs,
):
    """
    Compute the exact MSM barycenter average of time series.

    Parameters
    ----------
    X : np.ndarray
        Time series to average. Shape (n_cases, n_channels, n_timepoints).
    c : float, default=1.0
        Cost for Split/Merge operations.
    window : int, optional
        Window size constraint for the path.
    verbose : bool, default=False
        If True, print warning for large inputs.
    return_distances_to_center : bool, default=False
        If True, return distances from each input series to the center.
    return_cost : bool, default=False
        If True, return the total cost of the optimal path.

    Returns
    -------
    barycenter : np.ndarray
        The MSM mean time series (n_channels, n_timepoints).
    """
    if X.ndim == 3:
        _X = X
    elif X.ndim == 2:
        _X = X.reshape((X.shape[0], 1, X.shape[1]))
    else:
        raise ValueError("X must be a 2D or 3D array")

    if _X.shape[1] > 1:
        raise ValueError(
            "MSM Exact Mean is currently only implemented for univariate "
            "time series (n_channels=1)."
        )

    if _X.shape[0] > 6 and verbose:
        warnings.warn(
            "MSM exact mean is exponential in n_cases and may be slow.",
            RuntimeWarning,
            stacklevel=2,
        )
    barycenter_1d, cost = _msm_core(_X[:, 0, :], c=c, window=window)
    barycenter = barycenter_1d.reshape(1, -1)

    if return_distances_to_center:
        dists = np.zeros(len(_X))
        for i in range(len(_X)):
            dists[i] = msm_distance(_X[i], barycenter, c=c, window=window)

        if return_cost:
            return barycenter, dists, cost
        return barycenter, dists

    if return_cost:
        return barycenter, cost

    return barycenter


@njit(cache=True, fastmath=True)
def _msm_cost(new, x, y, c):
    """Calculate MSM operation cost."""
    if new < min(x, y) or new > max(x, y):
        return c + min(abs(new - x), abs(new - y))
    return c


@njit(cache=True, fastmath=True)
def _flat(coords, mults):
    """Flatten coordinates."""
    idx = 0
    for i in range(len(coords)):
        idx += coords[i] * mults[i]
    return idx


@njit(cache=True)
def _msm_core(X, c=1.0, window=None):
    """Execute Forward DP table fill and Traceback."""
    n_k, n_pts = X.shape
    d_vals = np.unique(X)
    n_d = len(d_vals)

    max_c = np.full(n_k, n_pts - 1, dtype=np.int32)
    # Maximum possible length of the mean path
    l_len = n_k * (n_pts - 1) + 1

    mults = np.zeros(n_k, dtype=np.int32)
    mults[0] = 1
    sum_len = max_c[0] + 1
    for i in range(1, n_k):
        mults[i] = sum_len
        sum_len *= max_c[i] + 1

    # DP Table (Sum_Coords * L_Len * N_Distinct)
    sz_s, sz_l, sz_c = 1, n_d, l_len * n_d
    table = np.full(sum_len * l_len * n_d, -1.0, dtype=np.float64)

    # Precompute first appearances for pruning
    app = np.full((n_d, n_k), 2147483647, dtype=np.int32)
    for s in range(n_d):
        for k in range(n_k):
            for z in range(n_pts):
                if X[k, z] == d_vals[s]:
                    app[s, k] = z
                    break

    q_cap = 200000  # Initial capacity
    q = np.zeros((q_cap, n_k), dtype=np.int32)
    vis = np.zeros(sum_len, dtype=np.bool_)

    origin = np.zeros(n_k, dtype=np.int32)
    f_orig = _flat(origin, mults)
    q[0], head, tail = origin, 0, 1
    vis[f_orig] = True

    for s in range(n_d):
        cost = 0.0
        for k in range(n_k):
            cost += abs(X[k, 0] - d_vals[s])
        table[f_orig * sz_c + s * sz_s] = cost

    # Forward Pass (BFS)
    while head < tail:
        curr = q[head]
        head += 1
        f_curr = _flat(curr, mults)

        valid = [i for i in range(n_k) if curr[i] > 0]
        preds = []
        if len(valid) > 0:
            for i in range(1, 1 << len(valid)):
                p = curr.copy()
                for j in range(len(valid)):
                    if (i >> j) & 1:
                        p[valid[j]] -= 1
                if window is None or (np.max(p) - np.min(p)) <= window:
                    preds.append(p)

        for len_idx in range(l_len):
            for s in range(n_d):
                # Pruning check
                valid_s = False
                for k in range(n_k):
                    if app[s, k] <= curr[k]:
                        valid_s = True
                        break
                if not valid_s:
                    continue

                idx = f_curr * sz_c + len_idx * sz_l + s * sz_s
                best = np.inf
                val_s = d_vals[s]

                for p in preds:
                    f_p = _flat(p, mults)

                    # Move/Split Cost
                    if len_idx > 0:
                        m_cost = 0.0
                        for k in range(n_k):
                            if curr[k] > p[k]:
                                m_cost += abs(X[k, curr[k]] - val_s)

                        base = f_p * sz_c + (len_idx - 1) * sz_l
                        for sp in range(n_d):
                            c_p = table[base + sp * sz_s]
                            if c_p < 0:
                                continue
                            sp_cost = 0.0
                            val_p = d_vals[sp]
                            for k in range(n_k):
                                if curr[k] == p[k]:
                                    sp_cost += _msm_cost(val_s, X[k, curr[k]], val_p, c)
                            best = min(best, c_p + m_cost + sp_cost)

                    # Merge Cost
                    c_me = table[f_p * sz_c + len_idx * sz_l + s * sz_s]
                    if c_me >= 0:
                        me_cost = 0.0
                        for k in range(n_k):
                            if curr[k] > p[k]:
                                me_cost += _msm_cost(
                                    X[k, curr[k]], X[k, curr[k] - 1], val_s, c
                                )
                        best = min(best, c_me + me_cost)

                if best < np.inf:
                    table[idx] = best

        # Queue Successors
        for k in range(n_k):
            if curr[k] < max_c[k]:
                nxt = curr.copy()
                nxt[k] += 1

                if window is not None and (np.max(nxt) - np.min(nxt)) > window:
                    continue

                f_nxt = _flat(nxt, mults)
                if not vis[f_nxt]:
                    vis[f_nxt] = True
                    if tail >= q_cap:
                        q_cap *= 2
                        new_q = np.zeros((q_cap, n_k), dtype=np.int32)
                        new_q[:tail] = q[:tail]
                        q = new_q
                    q[tail] = nxt
                    tail += 1

    # Traceback
    f_max = _flat(max_c, mults)
    min_cost, sl, ss = np.inf, 0, 0

    for s in range(n_d):
        for len_idx in range(l_len):
            val = table[f_max * sz_c + len_idx * sz_l + s * sz_s]
            if 0 <= val < min_cost:
                min_cost, sl, ss = val, len_idx, s

    path = np.zeros(sl + 1)
    path[sl] = d_vals[ss]
    idx_p = sl - 1
    cl, cs, cc = sl, ss, max_c.copy()

    while cl > 0 or np.any(cc > 0):
        f_cur = _flat(cc, mults)
        cur_cost = table[f_cur * sz_c + cl * sz_l + cs]

        valid = [i for i in range(n_k) if cc[i] > 0]
        found = False
        if len(valid) > 0:
            for i in range(1, 1 << len(valid)):
                p = cc.copy()
                for j in range(len(valid)):
                    if (i >> j) & 1:
                        p[valid[j]] -= 1

                if window is not None and (np.max(p) - np.min(p)) > window:
                    continue

                f_p = _flat(p, mults)
                v_s = d_vals[cs]

                # Check Move/Split
                if cl > 0:
                    m_cost = 0.0
                    for k in range(n_k):
                        if cc[k] > p[k]:
                            m_cost += abs(X[k, cc[k]] - v_s)

                    base = f_p * sz_c + (cl - 1) * sz_l
                    for sp in range(n_d):
                        c_p = table[base + sp * sz_s]
                        if c_p < 0:
                            continue
                        sp_cost = 0.0
                        v_p = d_vals[sp]
                        for k in range(n_k):
                            if cc[k] == p[k]:
                                sp_cost += _msm_cost(v_s, X[k, cc[k]], v_p, c)

                        if abs((c_p + m_cost + sp_cost) - cur_cost) < 1e-7:
                            cc, cl, cs = p, cl - 1, sp
                            path[idx_p] = d_vals[cs]
                            idx_p -= 1
                            found = True
                            break
                if found:
                    break

                # Check Merge
                c_me = table[f_p * sz_c + cl * sz_l + cs * sz_s]
                if c_me >= 0:
                    me_cost = 0.0
                    for k in range(n_k):
                        if cc[k] > p[k]:
                            me_cost += _msm_cost(X[k, cc[k]], X[k, cc[k] - 1], v_s, c)
                    if abs((c_me + me_cost) - cur_cost) < 1e-7:
                        cc, found = p, True
                        break
            if found:
                continue
        break

    return path, min_cost
