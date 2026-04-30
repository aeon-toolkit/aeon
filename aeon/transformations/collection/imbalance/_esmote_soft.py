"""Elastic SMOTE for time series classification using Soft-MSM distances."""

import numpy as np

from aeon.distances import msm_distance
from aeon.distances.elastic.soft import (
    soft_msm_alignment_matrix,
    soft_msm_distance,
    soft_msm_grad_x,
)


class ESMOTE_SOFT:
    """Elastic SMOTE with Soft-MSM synthetic generation.

    Extends the SMOTE oversampling idea to time series by using elastic distances
    to find neighbours and to interpolate synthetic samples between a minority-class
    series and one of its nearest neighbours.

    Three generation strategies are supported:

    ``"soft_path"``
        Use the row-normalised Soft-MSM alignment matrix ``A`` to compute an
        *expected* target for each index ``i``:
        ``y_expected[i] = (A[i,:] @ y) / sum(A[i,:])``.
        The synthetic is ``x + λ * (y_expected – x)``.

    ``"soft_barycenter"``
        Find the weighted Fréchet mean ``z`` that minimises
        ``(1–λ) * soft_msm(z, x) + λ * soft_msm(z, y)``
        by gradient descent, initialised from the soft-path interpolant.

    Parameters
    ----------
    k_neighbors : int, default=3
        Number of nearest neighbours to consider within the minority class.
    sampling_strategy : str, default="balance"
        Only ``"balance"`` is currently supported; generates exactly enough
        synthetic samples to match the majority-class count.
    distance : {"msm", "soft_msm"}, default="soft_msm"
        Distance used for nearest-neighbour search among minority samples.
        ``"soft_path"`` and ``"soft_barycenter"`` generation modes require
        ``"soft_msm"``.
    generation : {"hard_path", "soft_path", "soft_barycenter"},
        default="soft_path"
        Method used to create each synthetic sample.
    msm_cost : float, default=1.0
        The MSM split/merge penalty ``c``.  Must be non-negative.
    gamma : float, default=1.0
        Temperature parameter for Soft-MSM.  Smaller values produce sharper
        (closer to hard) alignments; larger values smooth the distribution over
        all alignments.  Must be positive.
    window : float or None, default=None
        Warping-window fraction passed to the distance functions.
    itakura_max_slope : float or None, default=None
        Itakura parallelogram slope limit passed to the distance functions.
    max_iter : int, default=50
        Maximum gradient-descent iterations for ``"soft_barycenter"`` generation.
    learning_rate : float, default=0.05
        Step size for gradient descent in ``"soft_barycenter"`` generation.
    tol : float, default=1e-6
        Convergence tolerance (RMS gradient norm) for ``"soft_barycenter"``.
    random_state : int or None, default=None
        Seed for the random number generator used to sample ``λ`` and to choose
        among tied neighbours.
    """

    def __init__(
        self,
        k_neighbors=3,
        sampling_strategy="balance",
        distance="soft_msm",
        generation="soft_path",
        msm_cost=1.0,
        gamma=1.0,
        window=None,
        itakura_max_slope=None,
        max_iter=50,
        learning_rate=0.05,
        tol=1e-6,
        random_state=None,
    ):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.distance = distance
        self.generation = generation
        self.msm_cost = msm_cost
        self.gamma = gamma
        self.window = window
        self.itakura_max_slope = itakura_max_slope
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y):
        """Return a rebalanced training set.

        Parameters
        ----------
        X : array-like of shape (n_cases, n_timepoints) or
            (n_cases, 1, n_timepoints)
            Univariate time series collection.
        y : array-like of shape (n_cases,)
            Class labels.  Exactly two distinct values are required.

        Returns
        -------
        X_out : ndarray, same dimensionality as input X
            Original samples followed by the synthetic minority samples.
        y_out : ndarray of shape (n_cases + n_synthetic,)
            Corresponding labels.
        """
        self._validate_params()

        X = np.asarray(X)
        y = np.asarray(y)

        input_was_2d = X.ndim == 2
        if input_was_2d:
            X = X[:, None, :]

        if X.ndim != 3:
            raise ValueError(
                "X must be 2D or 3D, with shape " "(n_cases, n_channels, n_timepoints)."
            )

        if X.shape[1] != 1:
            raise ValueError(
                "This ESMOTE implementation currently supports univariate series only."
            )

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("ESMOTE currently supports binary classification only.")

        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        X_min = X[y == minority_class]
        n_min = len(X_min)
        n_maj = np.sum(y == majority_class)
        n_to_generate = n_maj - n_min

        if n_to_generate <= 0:
            return (X[:, 0, :] if input_was_2d else X), y

        if n_min < 2:
            raise ValueError("At least two minority cases are required for ESMOTE.")

        rng = np.random.default_rng(self.random_state)
        neighbour_indices = self._find_minority_neighbours(X_min)

        X_synth = np.empty((n_to_generate, 1, X.shape[-1]), dtype=np.float64)

        # Round-robin over minority samples so every seed contributes equally.
        for i in range(n_to_generate):
            source_idx = i % n_min
            neighbour_idx = rng.choice(neighbour_indices[source_idx])

            x = X_min[source_idx, 0].astype(np.float64, copy=False)
            y_neighbour = X_min[neighbour_idx, 0].astype(np.float64, copy=False)

            X_synth[i, 0] = self._generate_synthetic(x, y_neighbour, rng)

        y_synth = np.full(n_to_generate, minority_class, dtype=y.dtype)

        X_out = np.concatenate([X.astype(np.float64, copy=False), X_synth], axis=0)
        y_out = np.concatenate([y, y_synth])

        if input_was_2d:
            X_out = X_out[:, 0, :]

        return X_out, y_out

    def _validate_params(self):
        valid_distances = {"soft_msm"}
        valid_generations = {"soft_path", "soft_barycenter"}

        if self.sampling_strategy != "balance":
            raise ValueError("Only sampling_strategy='balance' is currently supported.")

        if self.distance not in valid_distances:
            raise ValueError(
                f"distance must be one of {valid_distances}, got {self.distance!r}."
            )

        if self.generation not in valid_generations:
            raise ValueError(
                f"generation must be one of {valid_generations}, "
                f"got {self.generation!r}."
            )

        if self.generation in {"soft_path", "soft_barycenter"}:
            if self.distance != "soft_msm":
                raise ValueError(
                    "soft_path and soft_barycenter require distance='soft_msm'."
                )

        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")

        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")

        if self.msm_cost < 0:
            raise ValueError("msm_cost must be non-negative.")

    def _find_minority_neighbours(self, X_min):
        """Compute the k nearest neighbours within the minority class.

        Parameters
        ----------
        X_min : ndarray of shape (n_min, 1, n_timepoints)
            Minority-class samples.

        Returns
        -------
        indices : ndarray of shape (n_min, k)
            Indices of the k nearest neighbours for each sample.
            Self-distances are never filled in (they stay at ``np.inf``),
            so the self-index is always excluded.
        """
        n_cases = len(X_min)
        k = min(self.k_neighbors, n_cases - 1)

        # Upper-triangular fill; diagonal stays inf so self is never a neighbour.
        distances = np.full((n_cases, n_cases), np.inf)

        for i in range(n_cases):
            for j in range(i + 1, n_cases):
                dist = self._distance(X_min[i, 0], X_min[j, 0])
                distances[i, j] = dist
                distances[j, i] = dist

        return np.argsort(distances, axis=1)[:, :k]

    def _distance(self, x, y):
        """Compute the configured distance between two univariate series."""
        if self.distance == "msm":
            return msm_distance(x, y, c=self.msm_cost)

        if self.distance == "soft_msm":
            return soft_msm_distance(
                x,
                y,
                window=self.window,
                c=self.msm_cost,
                itakura_max_slope=self.itakura_max_slope,
                gamma=self.gamma,
            )

        raise RuntimeError("Unreachable distance branch.")

    def _generate_synthetic(self, x, y, rng):
        """Dispatch to the configured generation method."""
        if self.generation == "soft_path":
            return self._generate_soft_path(x, y, rng)

        if self.generation == "soft_barycenter":
            return self._generate_soft_barycenter(x, y, rng)

        raise RuntimeError("Unreachable generation branch.")

    def _generate_soft_path(self, x, y, rng, lam=None):
        if lam is None:
            lam = rng.random()

        result = soft_msm_alignment_matrix(
            x,
            y,
            window=self.window,
            c=self.msm_cost,
            itakura_max_slope=self.itakura_max_slope,
            gamma=self.gamma,
        )

        if isinstance(result, tuple):
            A = result[0]
        else:
            A = result

        A = np.asarray(A, dtype=np.float64)

        if A.shape != (len(x), len(y)):
            raise ValueError(
                "soft_msm_alignment_matrix must return shape "
                f"{(len(x), len(y))}, got {A.shape}."
            )

        row_sums = A.sum(axis=1)
        row_sums[row_sums == 0.0] = 1.0

        y_expected = (A @ y) / row_sums

        return x + lam * (y_expected - x)

    def _generate_soft_barycenter(self, x, y, rng):
        """Generate a synthetic series as the Soft-MSM two-series Fréchet mean.

        Finds the series ``z`` that minimises

            (1 - λ) * soft_msm(z, x) + λ * soft_msm(z, y)

        by gradient descent, initialised from the soft-path interpolant.
        """
        lam = rng.random()
        z = self._generate_soft_path(x, y, rng, lam=lam).astype(np.float64, copy=True)

        for _ in range(self.max_iter):
            grad_x = soft_msm_grad_x(
                z,
                x,
                window=self.window,
                c=self.msm_cost,
                itakura_max_slope=self.itakura_max_slope,
                gamma=self.gamma,
            )
            grad_y = soft_msm_grad_x(
                z,
                y,
                window=self.window,
                c=self.msm_cost,
                itakura_max_slope=self.itakura_max_slope,
                gamma=self.gamma,
            )

            # soft_msm_grad_x may return either grad or (grad, distance/objective).
            if isinstance(grad_x, tuple):
                grad_x = grad_x[0]
            if isinstance(grad_y, tuple):
                grad_y = grad_y[0]

            grad_x = np.asarray(grad_x, dtype=np.float64)
            grad_y = np.asarray(grad_y, dtype=np.float64)

            grad = (1.0 - lam) * grad_x + lam * grad_y

            grad_norm = np.linalg.norm(grad) / np.sqrt(len(grad))
            z -= self.learning_rate * grad

            if grad_norm < self.tol:
                break

        return z
