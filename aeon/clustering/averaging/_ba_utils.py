__maintainer__ = []

from typing import Optional, Union

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from aeon.distances import (
    adtw_alignment_path,
    ddtw_alignment_path,
    dtw_alignment_path,
    edr_alignment_path,
    erp_alignment_path,
    msm_alignment_path,
    pairwise_distance,
    shape_dtw_alignment_path,
    twe_alignment_path,
    wddtw_alignment_path,
    wdtw_alignment_path,
)


def _medoids(
    X: np.ndarray,
    precomputed_pairwise_distance: Union[np.ndarray, None] = None,
    distance: str = "dtw",
    **kwargs,
):
    if X.shape[0] < 1:
        return X

    if precomputed_pairwise_distance is None:
        precomputed_pairwise_distance = pairwise_distance(X, method=distance, **kwargs)

    x_size = X.shape[0]
    distance_matrix = np.zeros((x_size, x_size))
    for j in range(x_size):
        for k in range(x_size):
            distance_matrix[j, k] = precomputed_pairwise_distance[j, k]
    return X[np.argmin(sum(distance_matrix))]


def _get_init_barycenter(
    X: np.ndarray,
    init_barycenter: Union[np.ndarray, str],
    distance: str,
    precomputed_medoids_pw: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    if isinstance(init_barycenter, str):
        if init_barycenter not in ["mean", "medoids", "random"]:
            raise ValueError(
                "init_barycenter string is invalid. Please use one of the "
                "following: 'mean', 'medoids', 'random'"
            )
        if init_barycenter == "mean":
            return X.mean(axis=0)
        elif init_barycenter == "medoids":
            return _medoids(
                X,
                precomputed_pairwise_distance=precomputed_medoids_pw,
                distance=distance,
                **kwargs,
            )
        else:
            rng = check_random_state(random_state)
            return X[rng.choice(X.shape[0])]
    else:
        if not isinstance(init_barycenter, np.ndarray):
            raise ValueError(
                "init_barycenter parameter is invalid. It must either be "
                "a str or a np.ndarray"
            )
        if init_barycenter.shape != (X.shape[1], X.shape[2]):
            if init_barycenter.ndim == 1:
                return _get_init_barycenter(
                    X=X,
                    init_barycenter=init_barycenter.reshape(
                        1, init_barycenter.shape[0]
                    ),
                    distance=distance,
                    precomputed_medoids_pw=precomputed_medoids_pw,
                    random_state=random_state,
                    **kwargs,
                )
            raise ValueError(
                f"init_barycenter shape is invalid. Expected {(X.shape[1], X.shape[2])}"
                f" but got {init_barycenter.shape}"
            )

        return init_barycenter


VALID_BA_METRICS = [
    "dtw",
    "ddtw",
    "wdtw",
    "wddtw",
    "erp",
    "edr",
    "twe",
    "msm",
    "shape_dtw",
]


@njit(cache=True, fastmath=True)
def _get_alignment_path(
    center: np.ndarray,
    ts: np.ndarray,
    distance: str = "dtw",
    window: Union[float, None] = None,
    g: float = 0.0,
    epsilon: Union[float, None] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    independent: bool = True,
    c: float = 1.0,
    descriptor: str = "identity",
    reach: int = 30,
    warp_penalty: float = 1.0,
    transformation_precomputed: bool = False,
    transformed_x: Optional[np.ndarray] = None,
    transformed_y: Optional[np.ndarray] = None,
) -> tuple[list[tuple[int, int]], float]:
    if distance == "dtw":
        return dtw_alignment_path(ts, center, window)
    elif distance == "ddtw":
        return ddtw_alignment_path(ts, center, window)
    elif distance == "wdtw":
        return wdtw_alignment_path(ts, center, window, g)
    elif distance == "wddtw":
        return wddtw_alignment_path(ts, center, window, g)
    elif distance == "erp":
        return erp_alignment_path(ts, center, window, g)
    elif distance == "edr":
        return edr_alignment_path(ts, center, window, epsilon)
    elif distance == "twe":
        return twe_alignment_path(ts, center, window, nu, lmbda)
    elif distance == "msm":
        return msm_alignment_path(ts, center, window, independent, c)
    elif distance == "shape_dtw":
        return shape_dtw_alignment_path(
            ts,
            center,
            window=window,
            descriptor=descriptor,
            reach=reach,
            transformed_x=transformed_x,
            transformed_y=transformed_y,
            transformation_precomputed=transformation_precomputed,
        )
    elif distance == "adtw":
        return adtw_alignment_path(ts, center, window=window, warp_penalty=warp_penalty)
    else:
        # When numba version > 0.57 add more informative error with what method
        # was passed.
        raise ValueError("Distance parameter invalid")
