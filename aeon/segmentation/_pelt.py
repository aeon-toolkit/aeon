"""PELT (Pruned Exact Linear Time) Segmenter.

Optimal changepoint detection via dynamic programming with an exact pruning
rule that yields expected O(n) complexity. Supports L2 (mean change) and
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
from numba import njit

from aeon.segmentation.base import BaseSegmenter


@njit(cache=True)
def _cost_l2_numba(S: np.ndarray, SS: np.ndarray, start: int, end: int) -> float:
    length = end - start
    n_channels = S.shape[1]
    total = 0.0
    for c in range(n_channels):
        diff_s = S[end, c] - S[start, c]
        diff_ss = SS[end, c] - SS[start, c]
        dev = diff_ss - (diff_s * diff_s) / length
        if dev > 0.0:
            total += dev
    return total


@njit(cache=True)
def _cost_normal_numba(
    S: np.ndarray, SS: np.ndarray, start: int, end: int, eps: np.ndarray
) -> float:
    length = end - start
    n_channels = S.shape[1]
    total = 0.0
    for c in range(n_channels):
        diff_s = S[end, c] - S[start, c]
        diff_ss = SS[end, c] - SS[start, c]
        mean_c = diff_s / length
        var = diff_ss / length - mean_c * mean_c
        if var < 0.0:
            var = 0.0
        total += length * np.log(var + eps[c])
    return total


@njit(cache=True)
def _pelt_dp_loop(
    S: np.ndarray,
    SS: np.ndarray,
    grid: np.ndarray,
    pen: float,
    min_size: int,
    cost_type: int,
    eps: np.ndarray,
) -> np.ndarray:
    n_timepoints = S.shape[0] - 1
    F = np.full(n_timepoints + 1, np.inf, dtype=np.float64)
    F[0] = -pen
    last_cp = np.zeros(n_timepoints + 1, dtype=np.int64)
    expire = np.full(n_timepoints + 1, n_timepoints + 1, dtype=np.int64)

    candidates = np.zeros(n_timepoints + 1, dtype=np.int64)
    candidates[0] = 0
    n_candidates = 1

    cand_costs = np.zeros(n_timepoints + 1, dtype=np.float64)
    cand_evaluated = np.zeros(n_timepoints + 1, dtype=np.bool_)

    n_grid = len(grid)
    for g_idx in range(1, n_grid):
        t = grid[g_idx]

        # Drop candidates that have expired by current time t
        new_n_candidates = 0
        for i in range(n_candidates):
            tau = candidates[i]
            if expire[tau] > t:
                candidates[new_n_candidates] = tau
                new_n_candidates += 1
        n_candidates = new_n_candidates

        best_cost = np.inf
        best_tau = 0

        for i in range(n_candidates):
            tau = candidates[i]
            if t - tau >= min_size:
                if cost_type == 0:
                    c = _cost_l2_numba(S, SS, tau, t)
                else:
                    c = _cost_normal_numba(S, SS, tau, t, eps)
                cand_costs[i] = c
                cand_evaluated[i] = True
                total = F[tau] + c + pen
                if total < best_cost:
                    best_cost = total
                    best_tau = tau
            else:
                cand_evaluated[i] = False

        F[t] = best_cost
        last_cp[t] = best_tau

        # Pruning rule with delayed expiration for min_size >= 2
        if np.isfinite(F[t]):
            for i in range(n_candidates):
                if cand_evaluated[i]:
                    tau = candidates[i]
                    c = cand_costs[i]
                    if F[tau] + c >= F[t]:
                        exp_t = t + min_size
                        if exp_t < expire[tau]:
                            expire[tau] = exp_t

        if t >= min_size and np.isfinite(F[t]):
            candidates[n_candidates] = t
            n_candidates += 1

    # Backtrack optimal changepoint sequence
    curr = n_timepoints
    count = 0
    while last_cp[curr] != 0:
        count += 1
        curr = last_cp[curr]

    changepoints = np.empty(count, dtype=np.int64)
    curr = n_timepoints
    idx = count - 1
    while last_cp[curr] != 0:
        changepoints[idx] = last_cp[curr]
        curr = last_cp[curr]
        idx -= 1

    return changepoints


class PELTSegmenter(BaseSegmenter):
    r"""PELT (Pruned Exact Linear Time) changepoint detector.

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

    penalty : str, float, int, or None, default="bic"
        Penalty added for each additional changepoint.

        - ``"bic"`` : Bayesian Information Criterion. Under Killick et al.
          (2012, Section 2.1) [1]_, each changepoint introduces :math:`p = 2`
          parameters per channel (the changepoint location and the segment parameter).
          For ``cost="l2"``, the penalty is scaled by the channel noise variance
          :math:`\hat{\sigma}^2_c` estimated via the MAD of first differences:
          :math:`\beta = 2 \sum_c \hat{\sigma}^2_c \log(n)`.
          For ``cost="normal"``, both mean and variance are estimated per segment,
          yielding :math:`\beta = 2 \cdot n\_channels \log(n)`.
        - ``"aic"`` : Akaike Information Criterion:
          For ``cost="l2"``, :math:`\beta = 4 \sum_c \hat{\sigma}^2_c`.
          For ``cost="normal"``, :math:`\beta = 4 \cdot n\_channels`.
        - ``float`` or ``int`` : Manual penalty value (must be finite and >= 0).
        - ``None`` : No penalty (equivalent to 0).

    min_size : int, default=2
        Minimum number of observations between consecutive changepoints.
        Must be an integer >= 2.

    jump : int, default=1
        Subsample interval for candidate changepoint positions.
        ``jump=1`` considers every position; ``jump=k`` considers every
        k-th position. Larger values trade optimality for speed.
        Must be an integer >= 1.

    Notes
    -----
    The implementation uses prefix-sum arrays for O(d) cost evaluation per
    candidate (d = number of channels). Input data is centred per channel
    and cast to float64 before prefix-sum computation to mitigate
    floating-point cancellation.

    For the ``"normal"`` cost, a regularization constant epsilon relative to
    each channel's empirical variance (:math:`\epsilon_c = 10^{-8} \mathrm{Var}(X_c)`)
    is added inside the logarithm to ensure scale invariance and prevent numerical
    issues on constant-value segments (with a fallback to :math:`10^{-8}` if
    channel variance is zero).
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

        if np.isinf(X).any():
            raise ValueError("Input time series X contains infinite values.")

        if n_timepoints < 2 * self.min_size:
            return np.array([], dtype=np.intp)

        # Center per channel and cast to float64 for numerical stability.
        data = X.astype(np.float64)
        data = data - data.mean(axis=1, keepdims=True)

        data_T = data.T
        S = np.zeros((n_timepoints + 1, n_channels), dtype=np.float64)
        SS = np.zeros((n_timepoints + 1, n_channels), dtype=np.float64)
        np.cumsum(data_T, axis=0, out=S[1:])
        np.cumsum(data_T**2, axis=0, out=SS[1:])

        pen = self._compute_penalty(n_timepoints, n_channels, data)
        cost_type = 0 if self.cost == "l2" else 1

        if self.cost == "normal":
            eps = np.empty(n_channels, dtype=np.float64)
            for c in range(n_channels):
                chan_var = float(np.var(data[c]))
                eps[c] = 1e-8 * chan_var if chan_var > 0.0 else 1e-8
        else:
            eps = np.zeros(n_channels, dtype=np.float64)

        grid = np.arange(0, n_timepoints + 1, self.jump, dtype=np.int64)
        if grid[-1] != n_timepoints:
            grid = np.append(grid, n_timepoints)

        changepoints = _pelt_dp_loop(S, SS, grid, pen, self.min_size, cost_type, eps)
        return changepoints.astype(np.intp)

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
        return _cost_l2_numba(S, SS, start, end)

    @staticmethod
    def _cost_normal(
        S: np.ndarray, SS: np.ndarray, start: int, end: int, eps: float = 1e-8
    ) -> float:
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
        eps : float, default=1e-8
            Regularization constant.

        Returns
        -------
        float
            Segment cost.
        """
        n_channels = S.shape[1]
        eps_arr = np.full(n_channels, eps, dtype=np.float64)
        return _cost_normal_numba(S, SS, start, end, eps_arr)

    def _compute_penalty(
        self, n_timepoints: int, n_channels: int, data: np.ndarray | None = None
    ) -> float:
        """Compute the penalty value from *self.penalty*.

        Parameters
        ----------
        n_timepoints : int
            Length of the time series.
        n_channels : int
            Number of channels.
        data : np.ndarray or None, default=None
            Centered data array of shape ``(n_channels, n_timepoints)`` used to
            estimate channel noise variance for L2 penalty scaling.

        Returns
        -------
        float
            Penalty value (>= 0).
        """
        if self.penalty is None:
            return 0.0
        if not isinstance(self.penalty, str):
            return float(self.penalty)

        if self.cost == "l2":
            if data is not None and n_timepoints > 1:
                total_var = 0.0
                for c in range(n_channels):
                    diff = np.diff(data[c])
                    mad = float(np.median(np.abs(diff - np.median(diff))))
                    sigma = 1.4826 * mad / np.sqrt(2.0)
                    var_c = sigma**2
                    if var_c <= 0.0:
                        chan_var = float(np.var(data[c]))
                        var_c = max(1e-8 * chan_var, 1e-8)
                    total_var += var_c
            else:
                total_var = float(n_channels)

            # p = 2 parameters per changepoint per channel (Killick et al. 2012)
            p = 2.0 * total_var
            if self.penalty == "bic":
                return float(p * np.log(n_timepoints))
            return float(2.0 * p)
        else:
            # For normal cost, p = 2 * n_channels (mean and variance per channel)
            p = 2.0 * n_channels
            if self.penalty == "bic":
                return float(p * np.log(n_timepoints))
            return float(2.0 * p)

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

        if isinstance(self.penalty, bool):
            raise ValueError(f"penalty cannot be a boolean, got {self.penalty}.")
        if isinstance(self.min_size, bool):
            raise ValueError(f"min_size cannot be a boolean, got {self.min_size}.")
        if isinstance(self.jump, bool):
            raise ValueError(f"jump cannot be a boolean, got {self.jump}.")

        if self.penalty is None:
            pass
        elif isinstance(self.penalty, str):
            valid_penalties = ("bic", "aic")
            if self.penalty not in valid_penalties:
                raise ValueError(
                    f"Invalid penalty '{self.penalty}'. "
                    f"Must be one of {valid_penalties}, None, or a non-negative number."
                )
        elif isinstance(self.penalty, (int, float, np.number)):
            if not np.isfinite(self.penalty):
                raise ValueError(f"penalty must be finite, got {self.penalty}.")
            if self.penalty < 0:
                raise ValueError(
                    f"Manual penalty must be a non-negative number, got {self.penalty}."
                )
        else:
            raise ValueError(
                f"Manual penalty must be a non-negative number, string, or None, "
                f"got {self.penalty}."
            )

        if not isinstance(self.min_size, (int, np.integer)) or self.min_size < 2:
            raise ValueError(f"min_size must be an integer >= 2, got {self.min_size}.")

        if not isinstance(self.jump, (int, np.integer)) or self.jump < 1:
            raise ValueError(f"jump must be an integer >= 1, got {self.jump}.")

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
