"""Elastic SMOTE for time series classification using Soft-MSM alignments."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ESMOTE_SOFT"]

from collections import OrderedDict

import numpy as np

from aeon.distances import msm_distance
from aeon.distances.elastic.soft import soft_msm_alignment_matrix
from aeon.transformations.collection import BaseCollectionTransformer


class ESMOTE_SOFT(BaseCollectionTransformer):
    """Elastic SMOTE with hard MSM neighbours and Soft-MSM alignment generation.

    This is the minimal Soft-MSM variant of e-SMOTE. It keeps the neighbour
    search fixed to the paper setting, hard MSM with c=1 by default, and changes
    only the alignment-generation step.

    For each minority-class case x, a minority-class neighbour y is selected from
    the k nearest neighbours under hard MSM. A Soft-MSM alignment matrix A is
    then used to compute the expected aligned value from y for each index of x.

    The synthetic case is generated as

        x_new = x + lambda * direction

    where direction points from x towards its Soft-MSM expected alignment in y.
    This sign is chosen to match the negative gradient direction: the gradient of
    a pointwise match loss with respect to x points away from the aligned y value,
    so the interpolation step should move towards y.

    Parameters
    ----------
    k_neighbors : int, default=3
        Number of nearest neighbours to consider within the minority class.
    sampling_strategy : str, default="balance"
        Only "balance" is currently supported. This generates enough synthetic
        samples to match the majority-class count.
    msm_cost : float, default=1.0
        The MSM split/merge penalty c used for hard MSM neighbour search and
        Soft-MSM alignment.
    gamma : float, default=0.1
        Temperature parameter for Soft-MSM alignment. Smaller values produce
        sharper alignments; larger values smooth the distribution over alignments.
    window : float or None, default=None
        Warping-window fraction passed to the Soft-MSM alignment function.
    itakura_max_slope : float or None, default=None
        Itakura parallelogram slope limit passed to the Soft-MSM alignment
        function.
    random_state : int or None, default=None
        Seed for the random number generator used to sample lambda and
        neighbours.
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
        self,
        k_neighbors: int = 3,
        sampling_strategy="balance",
        msm_cost=1.0,
        gamma=0.1,
        window=None,
        itakura_max_slope=None,
        random_state=None,
    ):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.msm_cost = msm_cost
        self.gamma = gamma
        self.window = window
        self.itakura_max_slope = itakura_max_slope
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit the Soft-MSM e-SMOTE resampling strategy."""
        self._validate_params()
        self._rng = np.random.default_rng(self.random_state)

        if y is None:
            raise ValueError("ESMOTE_SOFT requires y during fit.")

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError(
                "ESMOTE_SOFT currently supports binary classification only."
            )

        self.sampling_strategy_ = self._get_sampling_strategy(y)

        return self

    def _transform(self, X, y=None):
        """Return a rebalanced collection and labels."""
        if y is None:
            raise ValueError("ESMOTE_SOFT requires y during transform.")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(
                "ESMOTE_SOFT expects X with shape "
                "(n_cases, n_channels, n_timepoints)."
            )

        if X.shape[1] != 1:
            raise ValueError("ESMOTE_SOFT currently supports univariate series only.")

        X_resampled = [X.astype(np.float64, copy=False)]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self._get_sampling_strategy(y).items():
            if n_samples == 0:
                continue

            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = X[target_class_indices]
            n_class = len(X_class)

            if n_class < 2:
                raise ValueError(
                    "At least two minority cases are required for ESMOTE_SOFT."
                )

            neighbour_indices = self._find_minority_neighbours(X_class)
            X_synth = np.empty((n_samples, 1, X.shape[-1]), dtype=np.float64)

            # Round-robin over minority samples, matching the paper's equal
            # contribution idea while allowing exact balancing when p is not
            # divisible by the number of minority cases.
            for i in range(n_samples):
                source_idx = i % n_class
                neighbour_idx = self._rng.choice(neighbour_indices[source_idx])

                x = X_class[source_idx, 0].astype(np.float64, copy=False)
                y_neighbour = X_class[neighbour_idx, 0].astype(np.float64, copy=False)

                X_synth[i, 0] = self._generate_soft_path(x, y_neighbour)

            y_synth = np.full(n_samples, class_sample, dtype=y.dtype)
            X_resampled.append(X_synth)
            y_resampled.append(y_synth)

        return np.vstack(X_resampled), np.hstack(y_resampled)

    def _get_sampling_strategy(self, y):
        """Return number of synthetic cases to generate for each class."""
        classes, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(classes, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)

        sampling_strategy = {
            key: n_sample_majority - value
            for key, value in target_stats.items()
            if key != class_majority
        }
        return OrderedDict(sorted(sampling_strategy.items()))

    def _validate_params(self):
        """Validate estimator parameters."""
        if self.sampling_strategy != "balance":
            raise ValueError("Only sampling_strategy='balance' is currently supported.")

        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")

        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")

        if self.msm_cost < 0:
            raise ValueError("msm_cost must be non-negative.")

    def _find_minority_neighbours(self, X_min):
        """Compute hard MSM nearest neighbours within the minority class.

        Parameters
        ----------
        X_min : ndarray of shape (n_minority, 1, n_timepoints)
            Minority-class samples.

        Returns
        -------
        indices : ndarray of shape (n_minority, k)
            Indices of the k nearest neighbours for each minority case.
        """
        n_cases = len(X_min)
        k = min(self.k_neighbors, n_cases - 1)

        distances = np.full((n_cases, n_cases), np.inf, dtype=np.float64)

        for i in range(n_cases):
            for j in range(i + 1, n_cases):
                dist = msm_distance(
                    X_min[i, 0],
                    X_min[j, 0],
                    c=self.msm_cost,
                )
                distances[i, j] = dist
                distances[j, i] = dist

        return np.argsort(distances, axis=1)[:, :k]

    def _generate_soft_path(self, x, y):
        """Generate a synthetic case using the Soft-MSM expected alignment."""
        lam = self._rng.random()

        result = soft_msm_alignment_matrix(
            x,
            y,
            window=self.window,
            c=self.msm_cost,
            itakura_max_slope=self.itakura_max_slope,
            gamma=self.gamma,
        )

        A = self._extract_alignment_matrix(result, len(x), len(y))

        row_sums = A.sum(axis=1)
        valid_rows = row_sums > 0.0

        direction = np.zeros_like(x, dtype=np.float64)

        # Expected aligned value in y for each index of x.
        #
        # The sign is important. The gradient with respect to x points from the
        # aligned y value towards x, so a synthetic interpolation step should
        # move in the opposite direction, i.e. towards the expected aligned y.
        y_expected = np.zeros_like(x, dtype=np.float64)
        y_expected[valid_rows] = (A[valid_rows] @ y) / row_sums[valid_rows]
        direction[valid_rows] = y_expected[valid_rows] - x[valid_rows]

        return x + lam * direction

    @staticmethod
    def _extract_alignment_matrix(result, n_timepoints_x, n_timepoints_y):
        """Extract the alignment matrix from the Soft-MSM return value."""
        expected_shape = (n_timepoints_x, n_timepoints_y)

        if isinstance(result, tuple):
            for item in result:
                candidate = np.asarray(item)
                if candidate.shape == expected_shape:
                    return candidate.astype(np.float64, copy=False)

            raise ValueError(
                "soft_msm_alignment_matrix returned a tuple, but none of its "
                f"entries had shape {expected_shape}."
            )

        A = np.asarray(result, dtype=np.float64)
        if A.shape != expected_shape:
            raise ValueError(
                "soft_msm_alignment_matrix must return shape "
                f"{expected_shape}, got {A.shape}."
            )

        return A

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
        """
        return {"k_neighbors": 1, "random_state": 0}
