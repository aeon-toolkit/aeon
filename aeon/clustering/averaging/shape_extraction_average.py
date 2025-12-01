import numpy as np
from sklearn.utils import check_random_state

from aeon.clustering._k_shape_original import _extract_shape
from aeon.clustering.averaging._shape_extraction import shape_extraction_average
from aeon.testing.data_generation import make_example_3d_numpy


def original_shape_extraction_average(
    X_ntc: np.ndarray,
    idx: np.ndarray,
    j: int,
    cur_center_ntc: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Compute the original K-Shape centroid (shape) for a single cluster j.

    This is a direct factorisation of the inner centroid-update logic from `_kshape`,
    without changing the core computations. It uses `_extract_shape` exactly as in
    the original implementation.

    Parameters
    ----------
    X_ntc : np.ndarray, shape (N, T, C)
        Time series data in (n_cases, n_timepoints, n_channels) format (N, T, C),
        as used in the original K-Shape code (time-major).
    idx : np.ndarray, shape (N,)
        Cluster assignment for each series, i.e., labels in {0, ..., k-1}.
    j : int
        The cluster index for which the centroid should be computed.
    cur_center_ntc : np.ndarray, shape (T, C)
        Current centroid for cluster j, in (T, C) format. This is the same
        slice that `_kshape` would use as `centroids[j]`.
    rng : np.random.RandomState
        Random number generator (same as passed into `_kshape` / `_extract_shape`).

    Returns
    -------
    new_center_ntc : np.ndarray, shape (T, C)
        Updated centroid for cluster j under the original K-Shape procedure.
    """
    X_ntc = np.asarray(X_ntc)
    idx = np.asarray(idx)
    cur_center_ntc = np.asarray(cur_center_ntc)

    if X_ntc.ndim != 3:
        raise ValueError(
            f"X_ntc must have shape (N, T, C), got {X_ntc.shape} (ndim={X_ntc.ndim})"
        )
    if cur_center_ntc.ndim != 2:
        raise ValueError(
            f"cur_center_ntc must have shape (T, C), got {cur_center_ntc.shape}"
        )

    N, T, C = X_ntc.shape
    if cur_center_ntc.shape != (T, C):
        raise ValueError(
            f"cur_center_ntc has shape {cur_center_ntc.shape}, " f"expected {(T, C)}"
        )

    # This mirrors the inner loop of `_kshape` for a fixed cluster j:
    #
    #   for d in range(x.shape[2]):
    #       centroids[j, :, d] = _extract_shape(
    #           idx,
    #           np.expand_dims(x[:, :, d], axis=2),
    #           j,
    #           np.expand_dims(centroids[j, :, d], axis=1),
    #           rng,
    #       )
    #
    new_center_ntc = np.zeros_like(cur_center_ntc)

    for d in range(C):
        # Extract dimension d as (N, T, 1)
        x_d = np.expand_dims(X_ntc[:, :, d], axis=2)  # (N, T, 1)
        cur_center_d = np.expand_dims(cur_center_ntc[:, d], axis=1)  # (T, 1)

        # `_extract_shape` already contains the original K-Shape averaging logic.
        new_center_d = _extract_shape(idx, x_d, j, cur_center_d, rng)  # (T,)
        new_center_ntc[:, d] = new_center_d

    return new_center_ntc


if __name__ == "__main__":
    rng = check_random_state(1)

    # Generate example data: (N, C, T)
    X = make_example_3d_numpy(
        n_cases=20,
        n_channels=1,
        n_timepoints=15,
        random_state=1,
        return_y=False,
    )

    # Original code works on (N, T, C)
    X_ntc = X.swapaxes(1, 2)  # (N, C, T) -> (N, T, C)
    N, T, C = X_ntc.shape

    # Fake a cluster assignment: all points belong to cluster 0
    idx = np.zeros(N, dtype=int)
    j = 0  # cluster index

    # Initial centre as mean (same for both implementations)
    # Original expects (T, C)
    cur_center_ntc = X_ntc.mean(axis=0)  # (T, C)

    # --- Original one-step K-Shape shape extraction ---
    orig_center_ntc = original_shape_extraction_average(
        X_ntc=X_ntc,
        idx=idx,
        j=j,
        cur_center_ntc=cur_center_ntc,
        rng=rng,
    )  # (T, C)

    # --- New shape_extraction_average implementation ---
    # This works on (N, C, T), so we pass X and init_centre in (C, T) format
    new_center_ct = shape_extraction_average(
        X,
        max_iters=1,
        tol=0.0,
        init_centre=cur_center_ntc.T,  # (T, C) -> (C, T)
        standardize=True,
        random_state=1,
        verbose=False,
    )  # (C, T)

    # Convert new centre into (T, C) for a direct comparison
    new_center_ntc = new_center_ct.T  # (C, T) -> (T, C)

    equal = np.allclose(orig_center_ntc, new_center_ntc)
    max_abs_diff = np.max(np.abs(orig_center_ntc - new_center_ntc))

    print(f"Centres equal (allclose): {equal}")
    print(f"Max abs difference: {max_abs_diff}")
