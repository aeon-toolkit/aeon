# tests/test_kshape_wrapper_equivalence.py
import itertools

import numpy as np
import pytest
from kshape.core import KShapeClusteringCPU, _ncc_c_3dim
from sklearn.utils import check_random_state

from aeon.clustering._original_kshape import TimeSeriesKShape  # noqa: E402

# ---------------------- helpers ----------------------


def sbd_ntc(x_ntc: np.ndarray, y_ntc: np.ndarray) -> float:
    """Shape-based distance for (T, C) arrays: 1 - max(NCC)."""
    return 1.0 - float(np.max(_ncc_c_3dim([x_ntc, y_ntc])))


def centroids_to_ntc_from_wrapper(C_kct: np.ndarray) -> np.ndarray:
    """Wrapper stores (k, C, T) -> convert to (k, T, C)."""
    return np.transpose(C_kct, (0, 2, 1))


def match_clusters_hungarian(Ca_ntc: np.ndarray, Cb_ntc: np.ndarray):
    """
    Find permutation p mapping clusters of A to B by minimising total SBD.
    Tries scipy Hungarian; falls back to brute-force (k! permutations) for small k.
    Returns an index array p of length k where p[i] = matched j.
    """
    k = Ca_ntc.shape[0]
    cost = np.empty((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            cost[i, j] = sbd_ntc(Ca_ntc[i], Cb_ntc[j])

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.empty(k, dtype=int)
        perm[row_ind] = col_ind
        return perm
    except Exception:
        # brute-force fallback (keep k small in tests if scipy unavailable)
        best_perm = None
        best_cost = np.inf
        for p in itertools.permutations(range(k)):
            total = sum(cost[i, p[i]] for i in range(k))
            if total < best_cost:
                best_cost = total
                best_perm = np.array(p, dtype=int)
        return best_perm


def relabel(labels: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Map wrapper labels (cluster i) to package labels (cluster perm[i])."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv[labels]


def compute_assignment_distances_ntc(
    X_ntc: np.ndarray, C_ntc: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Distance from each X to its assigned centroid (SBD)."""
    d = np.empty(X_ntc.shape[0], dtype=float)
    for i in range(X_ntc.shape[0]):
        d[i] = sbd_ntc(X_ntc[i], C_ntc[labels[i]])
    return d


# ---------------------- parametrised CPU equivalence tests ----------------------


@pytest.mark.parametrize(
    "n_cases,n_channels,n_timepoints,n_clusters,centroid_init,max_iter,seed",
    [
        (50, 1, 60, 3, "zero", 5, 123),
        (50, 1, 60, 3, "random", 5, 123),
        (32, 2, 40, 4, "random", 3, 7),
        (12, 3, 25, 2, "zero", 2, 2025),
    ],
)
def test_wrapper_matches_kshape_cpu(
    n_cases, n_channels, n_timepoints, n_clusters, centroid_init, max_iter, seed
):
    # 1) Build data with a dedicated RNG stream
    data_rng = check_random_state(seed)
    X_ntc = data_rng.randn(n_cases, n_timepoints, n_channels)

    # 2) Fit the package CPU model with a *fresh* RNG seeded identically
    pkg_rng = check_random_state(seed)  # fresh stream from the same seed
    np.random.set_state(pkg_rng.get_state())  # package relies on global NumPy RNG
    ksc = KShapeClusteringCPU(
        n_clusters,
        centroid_init=centroid_init,
        max_iter=max_iter,
        n_jobs=1,  # keep deterministic & fast
    )
    ksc.fit(X_ntc)
    labels_pkg = ksc.labels_.astype(int)
    C_pkg_ntc = ksc.centroids_  # (k, T, C)

    # 3) Fit the wrapper with the same seed (wrapper uses check_random_state internally)
    X_kct = np.transpose(X_ntc, (0, 2, 1))  # (N, C, T)
    ks_wrap = TimeSeriesKShape(
        n_clusters=n_clusters,
        centroid_init=centroid_init,
        max_iter=max_iter,
        n_jobs=1,
        random_state=seed,  # same seed as package
    )
    ks_wrap.fit(X_kct)
    labels_wrap = ks_wrap.labels_.astype(int)
    C_wrap_ntc = centroids_to_ntc_from_wrapper(ks_wrap.cluster_centers_)  # (k, T, C)

    # 4) Match clusters by centroids and compare everything
    perm = match_clusters_hungarian(C_wrap_ntc, C_pkg_ntc)

    # labels equality (after mapping)
    labels_wrap_mapped = relabel(labels_wrap, perm)
    assert labels_wrap_mapped.shape == labels_pkg.shape
    assert np.array_equal(labels_wrap_mapped, labels_pkg)

    # centroids closeness (after permutation)
    C_wrap_perm_ntc = C_wrap_ntc[perm]
    np.testing.assert_allclose(C_wrap_perm_ntc, C_pkg_ntc, rtol=1e-6, atol=1e-6)

    # predict consistency on train
    preds_pkg = ksc.predict(X_ntc)
    preds_wrap = ks_wrap.predict(X_kct)
    preds_wrap_mapped = relabel(preds_wrap, perm)
    assert np.array_equal(preds_wrap_mapped, preds_pkg)

    # inertia consistency (computed from SBD distances with package centroids/labels)
    d_train = compute_assignment_distances_ntc(X_ntc, C_pkg_ntc, labels_pkg)
    inertia_pkg_like = float(np.sum(d_train))
    assert np.isclose(ks_wrap.inertia_, inertia_pkg_like, rtol=1e-8, atol=1e-8)


# ---------------------- RNG behaviour tests ----------------------


@pytest.mark.parametrize("centroid_init", ["zero", "random"])
def test_random_state_reproducible(centroid_init):
    data_rng = check_random_state(42)
    X_ntc = data_rng.randn(20, 30, 1)
    X_kct = np.transpose(X_ntc, (0, 2, 1))

    m1 = TimeSeriesKShape(
        n_clusters=3,
        centroid_init=centroid_init,
        max_iter=3,
        n_jobs=1,
        random_state=123,
    )
    m2 = TimeSeriesKShape(
        n_clusters=3,
        centroid_init=centroid_init,
        max_iter=3,
        n_jobs=1,
        random_state=123,
    )

    m1.fit(X_kct)
    m2.fit(X_kct)

    C1_ntc = centroids_to_ntc_from_wrapper(m1.cluster_centers_)
    C2_ntc = centroids_to_ntc_from_wrapper(m2.cluster_centers_)
    perm = match_clusters_hungarian(C1_ntc, C2_ntc)

    # labels and centroids must match after permutation
    np.testing.assert_array_equal(relabel(m1.labels_, perm), m2.labels_)
    np.testing.assert_allclose(C1_ntc[perm], C2_ntc, rtol=1e-6, atol=1e-6)
