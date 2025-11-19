from collections.abc import Callable

import numpy as np
from numpy.random import RandomState

from aeon.distances import pairwise_distance


def _random_center_initialiser_indexes(
    *, X: np.ndarray, n_clusters: int, random_state: RandomState
) -> np.ndarray:
    return random_state.choice(X.shape[0], n_clusters, replace=False)


def _random_center_initialiser(
    *, X: np.ndarray, n_clusters: int, random_state: RandomState
) -> np.ndarray:
    return X[
        _random_center_initialiser_indexes(
            X=X, n_clusters=n_clusters, random_state=random_state
        )
    ]


def _first_center_initialiser_indexes(
    *, X: np.ndarray, n_clusters: int, random_state: RandomState, **kwargs
) -> np.ndarray:
    return np.arange(n_clusters)


def _first_center_initialiser(
    *, X: np.ndarray, n_clusters: int, random_state: RandomState
) -> np.ndarray:
    return X[
        _first_center_initialiser_indexes(
            X=X, n_clusters=n_clusters, random_state=random_state
        )
    ]


def _random_values_center_initialiser(
    *, X: np.ndarray, n_clusters: int, random_state: RandomState
):
    return random_state.rand(n_clusters, X.shape[1])


def _kmeans_plus_plus_center_initialiser_indexes(
    *,
    X: np.ndarray,
    n_clusters: int,
    random_state: RandomState,
    distance: str | Callable,
    distance_params: dict,
    n_jobs: int,
    return_distance_and_labels: bool = False,
    **kwargs,
) -> np.ndarray:
    n_samples = X.shape[0]
    initial_center_idx = random_state.randint(n_samples)
    indexes = [initial_center_idx]

    min_distances = pairwise_distance(
        X,
        X[[initial_center_idx]],
        method=distance,
        n_jobs=n_jobs,
        **distance_params,
    ).reshape(n_samples)

    labels = np.zeros(n_samples, dtype=int)

    for i in range(1, n_clusters):
        d = min_distances.copy()
        chosen = np.asarray(indexes, dtype=int)
        finite_mask = np.isfinite(d)
        if not np.any(finite_mask):
            candidates = np.setdiff1d(np.arange(n_samples), chosen, assume_unique=False)
            next_center_idx = random_state.choice(candidates)
            indexes.append(next_center_idx)

            new_distances = pairwise_distance(
                X,
                X[[next_center_idx]],
                method=distance,
                n_jobs=n_jobs,
                **distance_params,
            ).reshape(n_samples)

            closer_points = new_distances < min_distances
            min_distances[closer_points] = new_distances[closer_points]
            labels[closer_points] = i
            continue

        min_val = d[finite_mask].min()
        w = d - min_val
        w[~np.isfinite(w)] = 0.0
        w = np.clip(w, 0.0, None)
        w[chosen] = 0.0

        total = w.sum()
        if total <= 0.0:
            candidates = np.setdiff1d(np.arange(n_samples), chosen, assume_unique=False)
            next_center_idx = random_state.choice(candidates)
        else:
            p = w / total
            p = np.clip(p, 0.0, None)
            p_sum = p.sum()
            if p_sum <= 0.0:
                candidates = np.setdiff1d(
                    np.arange(n_samples), chosen, assume_unique=False
                )
                next_center_idx = random_state.choice(candidates)
            else:
                p = p / p_sum
                next_center_idx = random_state.choice(n_samples, p=p)

        indexes.append(next_center_idx)

        new_distances = pairwise_distance(
            X,
            X[[next_center_idx]],
            method=distance,
            n_jobs=n_jobs,
            **distance_params,
        ).reshape(n_samples)

        closer_points = new_distances < min_distances
        min_distances[closer_points] = new_distances[closer_points]
        labels[closer_points] = i

    if return_distance_and_labels:
        return np.array(indexes), labels, min_distances
    else:
        return np.array(indexes)


def _kmeans_plus_plus_center_initialiser(
    *,
    X: np.ndarray,
    n_clusters: int,
    random_state: RandomState,
    distance: str | Callable,
    distance_params: dict,
    n_jobs: int,
    return_distance_and_labels: bool = False,
    **kwargs,
) -> np.ndarray:
    indexes, labels, min_distances = _kmeans_plus_plus_center_initialiser_indexes(
        X=X,
        n_clusters=n_clusters,
        random_state=random_state,
        distance=distance,
        distance_params=distance_params,
        n_jobs=n_jobs,
        return_distance_and_labels=True,
    )
    if return_distance_and_labels:
        return X[indexes], labels, min_distances
    return X[indexes]


def _kmedoids_plus_plus_center_initialiser_indexes(
    *,
    X: np.ndarray,
    n_clusters: int,
    random_state: RandomState,
    distance: str | Callable,
    distance_params: dict,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    """K-medoids++ initialisation that returns indexes.

    This is a simpler variant of kmeans++ that uses minimum distances
    directly as probabilities without the sophisticated weighting scheme.
    """
    initial_center_idx = random_state.randint(X.shape[0])
    indexes = [initial_center_idx]

    for _ in range(1, n_clusters):
        pw_dist = pairwise_distance(
            X, X[indexes], method=distance, n_jobs=n_jobs, **distance_params
        )
        min_distances = pw_dist.min(axis=1)
        probabilities = min_distances / min_distances.sum()
        next_center_idx = random_state.choice(X.shape[0], p=probabilities)
        indexes.append(next_center_idx)

    return np.array(indexes)


def _kmedoids_plus_plus_center_initialiser(
    *,
    X: np.ndarray,
    n_clusters: int,
    random_state: RandomState,
    distance: str | Callable,
    distance_params: dict,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    """K-medoids++ initialisation that returns centers."""
    indexes = _kmedoids_plus_plus_center_initialiser_indexes(
        X=X,
        n_clusters=n_clusters,
        random_state=random_state,
        distance=distance,
        distance_params=distance_params,
        n_jobs=n_jobs,
    )
    return X[indexes]


def resolve_center_initialiser(
    init: str | np.ndarray,
    X: np.ndarray,
    n_clusters: int,
    random_state: RandomState,
    initialisers_dict: dict,
    distance: str | Callable | None = None,
    distance_params: dict | None = None,
    n_jobs: int = 1,
    custom_init_handlers: dict | None = None,
    use_indexes: bool = False,
) -> Callable | np.ndarray:
    """Resolve the center initialiser function or array from init parameter.

    Parameters
    ----------
    init : str or np.ndarray
        Initialisation method string or array of initial centers/indexes.
    X : np.ndarray
        Input data for validation.
    n_clusters : int
        Number of clusters.
    random_state : RandomState
        Random state for initialisation.
    initialisers_dict : dict
        Dictionary of available initialisers (CENTER_INITIALISERS or
            CENTER_INITIALISER_INDEXES).
    distance : str or Callable, optional
        Distance method (required for kmeans++/kmedoids++).
    distance_params : dict, optional
        Distance parameters (required for kmeans++/kmedoids++).
    n_jobs : int, default=1
        Number of jobs for parallel processing (used for kmeans++/kmedoids++).
    custom_init_handlers : dict, optional
        Dictionary of custom initialisation handlers for special cases (e.g.,
        {"build": handler}).
    use_indexes : bool, default=False
        If True, expects 1D arrays (indexes). If False, expects multi-dimensional
        arrays (centers).

    Returns
    -------
    Callable or np.ndarray
        Initialisation function or array.
    """
    valid_init_methods = ", ".join(sorted(initialisers_dict.keys()))

    if isinstance(init, str):
        # Check custom handlers first (e.g., "build" for k-medoids)
        if custom_init_handlers and init in custom_init_handlers:
            return custom_init_handlers[init]

        if init not in initialisers_dict:
            raise ValueError(
                f"The value provided for init: {init} is "
                f"invalid. The following are a list of valid init algorithms "
                f"strings: {valid_init_methods}. You can also pass a "
                f"np.ndarray of appropriate shape."
            )

        initialiser_func = initialisers_dict[init]
        if init in ("kmeans++", "kmedoids++"):
            # kmeans++ and kmedoids++ need additional parameters
            if distance is None or distance_params is None:
                raise ValueError(
                    f"distance and distance_params are required for {init} "
                    f"initialisation"
                )
            return lambda X: initialiser_func(
                X=X,
                n_clusters=n_clusters,
                random_state=random_state,
                distance=distance,
                distance_params=distance_params,
                n_jobs=n_jobs,
            )
        else:
            # random, first, random_values only need basic parameters
            return lambda X: initialiser_func(
                X=X,
                n_clusters=n_clusters,
                random_state=random_state,
            )
    else:
        if isinstance(init, np.ndarray):
            if len(init) != n_clusters:
                raise ValueError(
                    f"The value provided for init: {init} is "
                    f"invalid. Expected length {n_clusters}, got {len(init)}."
                )

            if use_indexes:
                if init.ndim != 1:
                    raise ValueError(
                        f"The value provided for init: {init} is "
                        f"invalid. Expected 1D array of shape ({n_clusters},), "
                        f"got {init.shape}."
                    )
                return init
            else:
                if init.ndim == 1:
                    raise ValueError(
                        f"The value provided for init: {init} is "
                        f"invalid. Expected multi-dimensional array of shape "
                        f"({n_clusters}, {X.shape[1]}, {X.shape[2]}), got {init.shape}."
                    )
                if init.shape[1:] != X.shape[1:]:
                    raise ValueError(
                        f"The value provided for init: {init} is "
                        f"invalid. Expected shape ({n_clusters}, {X.shape[1]}, "
                        f"{X.shape[2]}), got {init.shape}."
                    )
                return init.copy()
        else:
            raise ValueError(
                f"The value provided for init: {init} is "
                f"invalid. Expected a string or np.ndarray."
            )


CENTER_INITIALISERS = {
    "random": _random_center_initialiser,
    "first": _first_center_initialiser,
    "random_values": _random_values_center_initialiser,
    "kmeans++": _kmeans_plus_plus_center_initialiser,
    "kmedoids++": _kmedoids_plus_plus_center_initialiser,
}

CENTER_INITIALISER_INDEXES = {
    "random": _random_center_initialiser_indexes,
    "first": _first_center_initialiser_indexes,
    "kmeans++": _kmeans_plus_plus_center_initialiser_indexes,
    "kmedoids++": _kmedoids_plus_plus_center_initialiser_indexes,
}
