from collections.abc import Callable
from functools import partial

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
    *, X: np.ndarray, n_clusters: int, random_state: RandomState, **kwargs
):
    return random_state.rand(n_clusters, X.shape[-2], X.shape[-1])


def _kmeans_plus_plus_center_initialiser_indexes(
    *,
    X: np.ndarray,
    n_clusters: int,
    random_state: RandomState,
    distance: str | Callable,
    distance_params: dict,
    n_jobs: int = 1,
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
    n_jobs: int = 1,
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

    This uses a k-means++-style seeding procedure, but with medoids,
    and supports potentially negative distances by shifting the
    distance distribution to be non-negative.
    """
    n_samples = X.shape[0]
    initial_center_idx = random_state.randint(n_samples)
    indexes = [initial_center_idx]

    # Initial minimum distances to the first medoid
    min_distances = pairwise_distance(
        X,
        X[[initial_center_idx]],
        method=distance,
        n_jobs=n_jobs,
        **distance_params,
    ).reshape(n_samples)

    for _ in range(1, n_clusters):
        d = min_distances.copy()
        chosen = np.asarray(indexes, dtype=int)

        finite_mask = np.isfinite(d)
        if not np.any(finite_mask):
            candidates = np.setdiff1d(np.arange(n_samples), chosen, assume_unique=False)
            next_center_idx = random_state.choice(candidates)
        else:
            min_val = d[finite_mask].min()
            w = d - min_val

            w[~np.isfinite(w)] = 0.0

            w = np.clip(w, 0.0, None)

            w[chosen] = 0.0

            total = w.sum()
            if total <= 0.0:
                candidates = np.setdiff1d(
                    np.arange(n_samples), chosen, assume_unique=False
                )
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
    """Resolve the center initialiser function or array from init parameter."""
    valid_init_methods = ", ".join(sorted(initialisers_dict.keys()))

    if isinstance(init, str):
        # Custom handlers first (e.g., "build" for k-medoids)
        if custom_init_handlers and init in custom_init_handlers:
            # This is typically a bound method on the estimator (picklable)
            return custom_init_handlers[init]

        if init not in initialisers_dict:
            raise ValueError(
                f"The value provided for init: {init} is invalid. "
                f"The following are a list of valid init algorithms "
                f"strings: {valid_init_methods}. You can also pass a "
                f"np.ndarray of appropriate shape."
            )

        initialiser_func = initialisers_dict[init]

        # Initialisers that need distance info
        if init in ("kmeans++", "kmedoids++"):
            if distance is None or distance_params is None:
                raise ValueError(
                    f"distance and distance_params are required for {init} "
                    f"initialisation"
                )
            # Return a partial of the top-level function (picklable)
            return partial(
                initialiser_func,
                n_clusters=n_clusters,
                random_state=random_state,
                distance=distance,
                distance_params=distance_params,
                n_jobs=n_jobs,
            )

        # random_values doesn't need distance, just size + RNG
        if init == "random_values":
            return partial(
                initialiser_func,
                n_clusters=n_clusters,
                random_state=random_state,
            )

        # "random", "first", etc. – basic initialisers
        return partial(
            initialiser_func,
            n_clusters=n_clusters,
            random_state=random_state,
        )

    # ---- np.ndarray branch (indexes / centers) ----
    if isinstance(init, np.ndarray):
        if len(init) != n_clusters:
            raise ValueError(
                f"The value provided for init: {init} is invalid. "
                f"Expected length {n_clusters}, got {len(init)}."
            )

        if use_indexes:
            if init.ndim != 1:
                raise ValueError(
                    f"The value provided for init: {init} is invalid. "
                    f"Expected 1D array of shape ({n_clusters},), "
                    f"got {init.shape}."
                )
            if not np.issubdtype(init.dtype, np.integer):
                raise ValueError(
                    f"The value provided for init: {init} is invalid. "
                    f"Expected an array of integers, got {init.dtype}."
                )
            if init.min() < 0 or init.max() >= X.shape[0]:
                raise ValueError(
                    f"The value provided for init: {init} is invalid. "
                    f"Values must be in the range [0, {X.shape[0]})."
                )
            return init
        else:
            if init.ndim == 1:
                raise ValueError(
                    f"The value provided for init: {init} is invalid. "
                    f"Expected multi-dimensional array of shape "
                    f"({n_clusters}, {X.shape[1]}, {X.shape[2]}), "
                    f"got {init.shape}."
                )
            if init.shape[1:] != X.shape[1:]:
                raise ValueError(
                    f"The value provided for init: {init} is invalid. "
                    f"Expected shape ({n_clusters}, {X.shape[1]}, "
                    f"{X.shape[2]}), got {init.shape}."
                )
            return init.copy()

    raise ValueError(
        f"The value provided for init: {init} is invalid. "
        f"Expected a string or np.ndarray."
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
