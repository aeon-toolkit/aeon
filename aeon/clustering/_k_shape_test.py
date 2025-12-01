import numpy as np
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.clustering._k_shape_original import _ncc_c_3dim

from aeon.distances import sbd_distance, sbd_pairwise_distance

import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft


def pairwise_sbd_original(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Shape-Based Distance (SBD) between all series in X.

    SBD(x, y) = 1 - max(NCC(x, y)), where NCC is computed over all circular shifts.

    Parameters
    ----------
    X : np.ndarray
        Time series dataset. Shape:
        - (n_series, n_timestamps)        for univariate
        - (n_series, n_timestamps, n_channels) for multivariate

    Returns
    -------
    D : np.ndarray, shape (n_series, n_series)
        Symmetric matrix of SBD distances.
    """
    X = np.asarray(X, dtype=float)

    # Ensure shape is (N, T, C)
    if X.ndim == 2:
        # (N, T) -> (N, T, 1)
        X = X[..., None]
    elif X.ndim != 3:
        raise ValueError(
            f"X must have shape (N, T) or (N, T, C), got {X.shape} (ndim={X.ndim})"
        )

    n_series = X.shape[0]
    D = np.empty((n_series, n_series), dtype=float)

    for i in range(n_series):
        D[i, i] = 0.0
        x_i = X[i]
        for j in range(i + 1, n_series):
            x_j = X[j]

            ncc = _ncc_c_3dim([x_i, x_j])
            d = 1.0 - float(np.max(ncc))

            # Clamp tiny negative values from numerical noise
            if d < 0.0 and d > -1e-12:
                d = 0.0

            D[i, j] = d
            D[j, i] = d

    return D

if __name__ == "__main__":
    X_train = make_example_3d_numpy(10, 1, 10, random_state=1, return_y=False)
    our_sbd = sbd_pairwise_distance(X_train, standardize=False)

    X_train_swap = X_train.swapaxes(1,2)
    original = pairwise_sbd_original(X_train_swap)


    equal = np.allclose(original, our_sbd)
    stop = ""