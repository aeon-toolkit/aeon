"""PELT (Pruned Exact Linear Time) Segmenter.

Optimal changepoint detection via dynamic programming with an exact pruning
rule that yields expected O(n) complexity.  Supports L2 (mean change) and
Gaussian log-likelihood (mean + variance change) cost functions, automatic
penalty selection (BIC / AIC), and multivariate time series.

Notes
-----
Based on the work from [1]_.

References
----------
.. [1] Killick, R., Fearnhead, P. & Eckley, I.A. (2012).
   "Optimal detection of changepoints with a linear computational cost."
   Journal of the American Statistical Association, 107(500), 1590-1598.
   https://doi.org/10.1080/01621459.2012.737745
"""

__maintainer__ = []
__all__ = ["PELTSegmenter"]

import numpy as np

from aeon.segmentation.base import BaseSegmenter


class PELTSegmenter(BaseSegmenter):
    """PELT (Pruned Exact Linear Time) changepoint detector.

    Detects an unknown number of changepoints in a time series by minimising
    a penalised cost function using dynamic programming with an exact pruning
    rule that yields expected O(n) complexity [1]_.

    The algorithm guarantees that the detected changepoints are the globally
    optimal solution under the chosen cost and penalty.

    Parameters
    ----------
    cost : str, default="l2"
        Cost function for each segment.

        - ``"l2"`` : Sum of squared deviations from the segment mean.
          Detects changes in mean.
        - ``"normal"`` : Negative Gaussian log-likelihood (mean + variance).
          Detects changes in mean and/or variance.

    penalty : str or float, default="bic"
        Penalty added for each additional changepoint.

        - ``"bic"`` : Bayesian Information Criterion,
          ``penalty = p * log(n_timepoints)`` where ``p`` is the number of
          free parameters per segment (``n_channels`` for L2,
          ``2 * n_channels`` for normal).
        - ``"aic"`` : Akaike Information Criterion, ``penalty = 2 * p``.
        - ``float`` : Manual penalty value (must be > 0).

    min_size : int, default=2
        Minimum number of observations between consecutive changepoints.
        Must be >= 2.

    jump : int, default=1
        Subsample interval for candidate changepoint positions.
        ``jump=1`` considers every position; ``jump=k`` considers every
        k-th position.  Larger values trade optimality for speed.

    Notes
    -----
    The implementation uses prefix-sum arrays for O(d) cost evaluation per
    candidate (d = number of channels).  Input data is centred per channel
    and cast to float64 before prefix-sum computation to mitigate
    floating-point cancellation.

    For the ``"normal"`` cost, an epsilon of 1e-8 is added inside the
    logarithm to prevent numerical issues on constant-value segments.
    Setting ``min_size >= 10`` is recommended when using ``cost="normal"``
    to prevent spurious short segments resulting from empirical variance
    underestimation.

    References
    ----------
    .. [1] Killick, R., Fearnhead, P. & Eckley, I.A. (2012).
       "Optimal detection of changepoints with a linear computational cost."
       Journal of the American Statistical Association, 107(500), 1590-1598.
       https://doi.org/10.1080/01621459.2012.737745

    Examples
    --------
    >>> from aeon.segmentation import PELTSegmenter
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    >>> pelt = PELTSegmenter(cost="l2", penalty="bic")
    >>> found_cps = pelt.fit_predict(X)
    """

    _tags = {
        "fit_is_empty": True,
        "capability:multivariate": True,
        "returns_dense": True,
    }

    def __init__(
        self,
        cost: str = "l2",
        penalty="bic",
        min_size: int = 2,
        jump: int = 1,
    ):
        self.cost = cost
        self.penalty = penalty
        self.min_size = min_size
        self.jump = jump
        super().__init__(axis=1, n_segments=None)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Segment *X* using the PELT algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_timepoints)
            Time series to segment.

        Returns
        -------
        np.ndarray
            Sorted array of changepoint indices.
        """
        self._validate_params()

        n_channels, n_timepoints = X.shape

        if n_timepoints < 2 * self.min_size:
            return np.array([], dtype=np.intp)

        # --- prefix-sum preparation ---
        # Centre per channel and promote to float64 for stability.
        data = X.astype(np.float64)
        data = data - data.mean(axis=1, keepdims=True)

        # S[t, c]  = sum_{i=0}^{t-1} data[c, i]   (S[0, :] = 0)
        # SS[t, c] = sum_{i=0}^{t-1} data[c, i]^2  (SS[0, :] = 0)
        data_T = data.T  # (n_timepoints, n_channels)
        S = np.zeros((n_timepoints + 1, n_channels), dtype=np.float64)
        SS = np.zeros((n_timepoints + 1, n_channels), dtype=np.float64)
        np.cumsum(data_T, axis=0, out=S[1:])
        np.cumsum(data_T**2, axis=0, out=SS[1:])

        # --- penalty ---
        pen = self._compute_penalty(n_timepoints, n_channels)

        # --- admissible changepoint grid ---
        grid = list(range(0, n_timepoints + 1, self.jump))
        if grid[-1] != n_timepoints:
            grid.append(n_timepoints)

        # --- PELT dynamic programming ---
        F: dict[int, float] = {0: -pen}
        cp: dict[int, int] = {0: 0}
        candidates: set[int] = {0}

        cost_fn = self._cost_l2 if self.cost == "l2" else self._cost_normal

        for t in grid[1:]:
            best_cost = np.inf
            best_tau = 0
            seg_costs: dict[int, float] = {}

            for tau in candidates:
                if t - tau >= self.min_size:
                    c = cost_fn(S, SS, tau, t)
                    seg_costs[tau] = c
                    total = F[tau] + c + pen
                    if total < best_cost:
                        best_cost = total
                        best_tau = tau

            F[t] = best_cost
            cp[t] = best_tau

            # Pruning (K = 0 for L2 / normal costs):
            # keep tau iff F[tau] + C(tau..t) <= F[t]
            new_candidates: set[int] = set()
            for tau in candidates:
                if tau in seg_costs:
                    if F[tau] + seg_costs[tau] <= F[t]:
                        new_candidates.add(tau)
                else:
                    # segment too short to evaluate — keep for future t
                    new_candidates.add(tau)
            new_candidates.add(t)
            candidates = new_candidates

        # --- backtrack ---
        changepoints: list[int] = []
        t = n_timepoints
        while cp.get(t, 0) != 0:
            changepoints.append(cp[t])
            t = cp[t]

        changepoints.sort()
        return np.array(changepoints, dtype=np.intp)

    # -----------------------------------------------------------------
    # Cost functions
    # -----------------------------------------------------------------

    @staticmethod
    def _cost_l2(S: np.ndarray, SS: np.ndarray, start: int, end: int) -> float:
        r"""L2 cost: sum of squared deviations from segment mean.

        .. math::
            \mathcal{C} = \sum_{c} \left[
                (SS_{end,c} - SS_{start,c})
                - \frac{(S_{end,c} - S_{start,c})^{2}}{L}
            \right]

        Parameters
        ----------
        S : np.ndarray
            Prefix sums, shape ``(n+1, n_channels)``.
        SS : np.ndarray
            Prefix sums of squares, shape ``(n+1, n_channels)``.
        start : int
            Segment start (exclusive).
        end : int
            Segment end (inclusive).

        Returns
        -------
        float
            Segment cost.
        """
        length = end - start
        diff_s = S[end] - S[start]
        diff_ss = SS[end] - SS[start]
        return float(np.sum(np.maximum(diff_ss - diff_s**2 / length, 0.0)))

    @staticmethod
    def _cost_normal(S: np.ndarray, SS: np.ndarray, start: int, end: int) -> float:
        r"""Gaussian log-likelihood cost (mean + variance change).

        .. math::
            \mathcal{C} = \sum_{c} L \, \log(\hat{\sigma}^{2}_{c} + \epsilon)

        where :math:`\hat{\sigma}^{2}_{c}` is the MLE variance for channel
        *c* on the segment :math:`(start, end]`.

        Parameters
        ----------
        S : np.ndarray
            Prefix sums, shape ``(n+1, n_channels)``.
        SS : np.ndarray
            Prefix sums of squares, shape ``(n+1, n_channels)``.
        start : int
            Segment start (exclusive).
        end : int
            Segment end (inclusive).

        Returns
        -------
        float
            Segment cost.
        """
        length = end - start
        diff_s = S[end] - S[start]
        diff_ss = SS[end] - SS[start]
        var = np.maximum(diff_ss / length - (diff_s / length) ** 2, 0.0)
        return float(np.sum(length * np.log(var + 1e-8)))

    # -----------------------------------------------------------------
    # Penalty computation
    # -----------------------------------------------------------------

    def _compute_penalty(self, n_timepoints: int, n_channels: int) -> float:
        """Compute the penalty value from *self.penalty*.

        Parameters
        ----------
        n_timepoints : int
            Length of the time series.
        n_channels : int
            Number of channels.

        Returns
        -------
        float
            Penalty value (> 0).
        """
        # Number of free parameters per segment.
        params_per_channel = 1 if self.cost == "l2" else 2
        p = n_channels * params_per_channel

        if isinstance(self.penalty, str):
            if self.penalty == "bic":
                return p * np.log(n_timepoints)
            # "aic"
            return 2.0 * p
        return float(self.penalty)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate_params(self) -> None:
        """Validate constructor parameters.

        Raises
        ------
        ValueError
            If any parameter is out of its valid range.
        """
        valid_costs = ("l2", "normal")
        if self.cost not in valid_costs:
            raise ValueError(
                f"Invalid cost '{self.cost}'. Must be one of {valid_costs}."
            )

        if isinstance(self.penalty, str):
            valid_penalties = ("bic", "aic")
            if self.penalty not in valid_penalties:
                raise ValueError(
                    f"Invalid penalty '{self.penalty}'. "
                    f"Must be one of {valid_penalties} or a positive float."
                )
        elif not (isinstance(self.penalty, (int, float)) and self.penalty > 0):
            raise ValueError(
                f"Manual penalty must be a positive number, " f"got {self.penalty}."
            )

        if not isinstance(self.min_size, int) or self.min_size < 2:
            raise ValueError(f"min_size must be an integer >= 2, got {self.min_size}.")

        if not isinstance(self.jump, int) or self.jump < 1:
            raise ValueError(f"jump must be an integer >= 1, got {self.jump}.")

    # -----------------------------------------------------------------
    # Estimator test interface
    # -----------------------------------------------------------------

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.
            If no special parameters are defined for a value, will return
            ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test
            instance, i.e., ``MyClass(**params)`` or
            ``MyClass(**params[i])`` creates a valid test instance.
        """
        params1 = {"cost": "l2", "penalty": "bic", "min_size": 2}
        params2 = {"cost": "normal", "penalty": 10.0, "min_size": 3, "jump": 2}
        return [params1, params2]
