"""DAMP anomaly detector."""

__maintainer__ = ["JayeshSuryavanshi"]
__all__ = ["DAMP"]

import warnings

import numpy as np
from numba import njit

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.anomaly_detection.series.distance_based._madrid import _mass, _next_pow2
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD
from aeon.utils.numba.stats import std


class DAMP(BaseSeriesAnomalyDetector):
    """DAMP streaming left-discord anomaly detector.

    DAMP (Discord Aware Matrix Profile) [1]_ finds the discords of a single
    subsequence length in a way that scales to very long series and to fast-arriving
    data streams. It is the single-length search that MADRID runs for every
    candidate length.

    The left matrix profile value of a subsequence is the z-normalised Euclidean
    distance to its nearest neighbour among the subsequences that end before it
    starts, so only past data is consulted. A left discord is a subsequence with a
    large left matrix profile value: nothing similar has been seen before. The top
    ``n_discords`` discords are selected greedily, as in the reference
    implementation: take the subsequence with the largest value, exclude every start
    position within ``window_size // 2`` of it, and repeat.

    DAMP does not compute the left matrix profile exactly everywhere. The backward
    search for a subsequence's nearest neighbour stops as soon as a neighbour closer
    than the current discord threshold is found, and a forward search over the next
    ``lookahead`` points skips future subsequences that are already known to lie
    no further than the threshold from an earlier one. The threshold never exceeds
    the final ``n_discords``-th discord score, so no skipped subsequence can score
    higher than the returned discords: the discord scores and locations are those of
    the exact left matrix profile, up to floating-point rounding. When subsequences
    tie exactly, a different tied location can be returned, which can also change
    the discords selected after it. The other entries of the profile are
    approximations.

    :meth:`predict` builds pointwise scores from exactly those discords: each
    discord paints its score onto the ``window_size`` points its subsequence covers,
    combined with the maximum where discords overlap, and every other point scores
    zero. Scores are raw z-normalised Euclidean distances; a single subsequence
    length is used, so no length normalisation is applied. Higher scores indicate
    more anomalous points. The approximate left matrix profile and the discord
    details are available via :meth:`predict_discords`.

    Points before ``train_test_split`` form a warm-up (training) region that is only
    used as reference history and is always scored zero.

    Parameters
    ----------
    window_size : int, default=50
        Length of the subsequences compared. Must be at least 4. The reference
        implementation suggests roughly 50 to 90 percent of a typical period.
    n_discords : int, default=1
        Number of top discords to find. Must be at least 1. Fewer are returned only
        when the greedy exclusion zones cover the whole region after
        ``train_test_split`` before ``n_discords`` discords are found. A larger value
        lowers the pruning threshold, so the search gets slower as it grows.
    train_test_split : int, float or None, default=None
        Location of the split point between the warm-up (training) region and the
        region searched for anomalies. An ``int`` is used directly as the split
        index. A ``float`` in ``(0, 1)`` is interpreted as a fraction of the series
        length. If ``None``, a warm-up of ``max(4 * window_size, len(X) // 5)``
        points is used, capped at ``len(X) - window_size``. The split must satisfy
        ``window_size <= split <= len(X) - window_size``. A warning is raised if the
        warm-up is shorter than ``4 * window_size`` points.
    lookahead : int or None, default=None
        Number of points after each processed subsequence searched by the forward
        pruning step. If ``None``, the reference implementation's default of
        ``16 * window_size`` rounded up to a power of two is used. ``0`` disables
        forward pruning, which gives a purely online algorithm. Other values are
        rounded up to the next power of two. The lookahead affects the speed, not
        the number or the scores of the discords found.

    See Also
    --------
    MADRID : Runs DAMP over a range of subsequence lengths.

    References
    ----------
    .. [1] Yue Lu, Renjie Wu, Abdullah Mueen, Maria A. Zuluaga and Eamonn Keogh,
           "Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions
           of Datapoints and Ultra-fast Arriving Data Streams," Proceedings of the
           28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD),
           Washington DC, USA, 2022, pp. 1173-1182.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.distance_based import DAMP
    >>> rng = np.random.default_rng(2)
    >>> X = np.sin(np.linspace(0, 20 * np.pi, 400)) + rng.normal(0, 0.05, 400)
    >>> X[300:330] += rng.normal(0, 1, 30)  # inject an anomalous noisy segment
    >>> detector = DAMP(window_size=16, train_test_split=100)
    >>> scores = detector.fit_predict(X)
    >>> bool(300 <= int(np.argmax(scores)) < 330)
    True
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        window_size=50,
        n_discords=1,
        train_test_split=None,
        lookahead=None,
    ):
        self.window_size = window_size
        self.n_discords = n_discords
        self.train_test_split = train_test_split
        self.lookahead = lookahead

        super().__init__(axis=1)

    def _predict(self, X):
        _, scores, locations, _ = self._run_damp(X)
        n = X.squeeze().shape[0]
        return self._to_pointwise_scores(scores, locations, int(self.window_size), n)

    def _run_damp(self, X):
        X = np.atleast_1d(np.asarray(X).squeeze())
        if X.ndim != 1:
            raise ValueError("DAMP only supports univariate series")
        n = X.shape[0]

        if self.window_size < 4:
            raise ValueError("window_size must be at least 4")
        elif self.n_discords < 1:
            raise ValueError(f"n_discords {self.n_discords} must be at least 1")
        elif self.lookahead is not None and self.lookahead < 0:
            raise ValueError(f"lookahead {self.lookahead} must be None or at least 0")
        m = int(self.window_size)
        if n < 2 * m:
            raise ValueError(
                f"Series length of X {n} must be at least double window_size {m}"
            )

        split = self._resolve_split(n)
        if split < m or split > n - m:
            raise ValueError(
                f"train_test_split resolved to {split}, but it must lie in "
                f"[window_size, len(X) - window_size] = [{m}, {n - m}]"
            )
        if split < 4 * m:
            warnings.warn(
                f"train_test_split {split} is less than four times window_size {m}. "
                "A warm-up region of at least 4 * window_size points is recommended, "
                "otherwise there may be false positives early in the test region.",
                stacklevel=2,
            )

        ts = np.ascontiguousarray(X, dtype=np.float64)
        if _has_near_constant_window(ts, m):
            warnings.warn(
                "There is a region close to constant that will cause the results to "
                "be unstable. It is suggested to delete the constant region or try "
                "again with a longer window_size.",
                stacklevel=2,
            )

        if self.lookahead is None:
            lookahead = 1 << (16 * m - 1).bit_length()
        elif self.lookahead == 0:
            lookahead = 0
        else:
            lookahead = 1 << (int(self.lookahead) - 1).bit_length()

        return _damp_top_k(ts, m, split, int(self.n_discords), lookahead)

    @staticmethod
    def _to_pointwise_scores(scores, locations, window_size, n):
        """Pointwise scores built only from the discords DAMP identified.

        Pruned and early-abandoned entries of the left matrix profile are not exact,
        so only the top discords are used: each one paints its score onto the points
        its subsequence covers, combined with the maximum where discords overlap.
        Points covered by no discord score zero.
        """
        point_scores = np.zeros(n)
        for score, loc in zip(scores, locations):
            start = int(loc)
            end = min(start + window_size, n)
            point_scores[start:end] = np.maximum(point_scores[start:end], score)
        return point_scores

    def predict_discords(self, X):
        """Run DAMP and return the discords and the approximate left matrix profile.

        Unlike :meth:`predict`, which reduces the result to one score per time
        point, this exposes everything the algorithm computes, mirroring the
        reference implementation's output.

        Parameters
        ----------
        X : np.ndarray
            One-dimensional time series.

        Returns
        -------
        dict
            ``scores``: the top discord scores in descending order;
            ``locations``: the start index of each discord, in the same order. Fewer
            than ``n_discords`` are returned only when the exclusion zones cover the
            whole region after the split first;
            ``left_matrix_profile``: the approximate left matrix profile, of length
            ``len(X) - window_size + 1`` and zero before the split. Entries skipped
            by forward pruning repeat the previous value and entries whose backward
            search stopped early hold an upper bound, so only the values at the
            returned locations are guaranteed exact;
            ``pruning_rate``: the fraction of subsequences after the split that were
            skipped by forward pruning;
            ``best_interval``: ``(start, end)`` of the top discord, matching the
            key returned by MADRID.
        """
        left_mp, scores, locations, pruning_rate = self._run_damp(X)
        start = int(locations[0])
        return {
            "scores": scores,
            "locations": locations,
            "left_matrix_profile": left_mp,
            "pruning_rate": float(pruning_rate),
            "best_interval": (start, start + int(self.window_size)),
        }

    def _resolve_split(self, n):
        split = self.train_test_split
        if split is None:
            m = int(self.window_size)
            return min(max(4 * m, n // 5), n - m)
        if isinstance(split, float) and 0.0 < split < 1.0:
            return int(round(n * split))
        return int(split)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid
            test instance.
        """
        return {"window_size": 4, "n_discords": 2, "train_test_split": 16}


@njit(cache=True)
def _has_near_constant_window(ts, m):
    for i in range(len(ts) - m + 1):
        if std(ts[i : i + m]) <= AEON_NUMBA_STD_THRESHOLD:
            return True
    return False


@njit(cache=True)
def _backward_search(ts, m, i, threshold):
    """Left matrix profile value at ``i`` and whether the search was exhaustive.

    Uses the doubling-prefix search of MADRID's DAMP kernel. The search stops at
    the first window containing a neighbour closer than ``threshold``, which leaves
    an upper bound below the threshold; otherwise it reaches the start of the series
    and the value is exact.
    """
    candidate = np.inf
    exact = False
    prefix = 2 ** _next_pow2(m)
    endpoint = i
    while candidate >= threshold:
        if endpoint - prefix <= 0:
            value = np.nanmin(_mass(ts[:endpoint], ts[i : i + m]))
            if value < candidate:
                candidate = value
            exact = True
            break
        value = np.nanmin(_mass(ts[endpoint - prefix : endpoint], ts[i : i + m]))
        if value < candidate:
            candidate = value
        if value < threshold:
            break
        endpoint = endpoint - prefix + m - 1
        prefix = 2 * prefix
    if candidate == np.inf:
        return 0.0, False
    return candidate, exact


@njit(cache=True)
def _greedy_separated(positions, values, k, exclusion, blocked):
    """Greedy top-``k`` whose start positions are more than ``exclusion`` apart.

    Repeatedly takes the largest remaining value (the earliest position on ties) and
    discards every position within ``exclusion`` of it. ``blocked`` is an all-False
    work array indexed by position; it is left all-False on return.
    """
    order = np.argsort(-values, kind="mergesort")
    scores = np.empty(k, dtype=np.float64)
    locations = np.empty(k, dtype=np.int64)
    found = 0
    for t in order:
        p = positions[t]
        if blocked[p]:
            continue
        scores[found] = values[t]
        locations[found] = p
        found += 1
        if found == k:
            break
        blocked[max(p - exclusion, 0) : p + exclusion + 1] = True
    for s in range(found):
        p = locations[s]
        blocked[max(p - exclusion, 0) : p + exclusion + 1] = False
    return scores[:found].copy(), locations[:found].copy()


@njit(cache=True)
def _damp_top_k(ts, m, split, n_discords, lookahead):
    """DAMP approximate left matrix profile and its exact top-k left discords.

    ``threshold`` never exceeds the final k-th discord score, so no skipped or
    early-abandoned position can score higher than the returned discords. With
    several discords, forward pruning waits for a positive threshold: while it is
    zero, a subsequence at distance zero may still be one of the discords.
    """
    n = len(ts)
    n_sub = n - m + 1
    radius = m // 2
    left_mp = np.zeros(n_sub)
    skipped = np.zeros(n_sub, dtype=np.bool_)
    exact = np.zeros(n_sub, dtype=np.bool_)
    blocked = np.zeros(n_sub, dtype=np.bool_)
    buffer_pos = np.empty(n_sub, dtype=np.int64)
    buffer_val = np.empty(n_sub, dtype=np.float64)
    buffer_len = 0
    min_span = (n_discords - 1) * (2 * radius + 1)
    threshold = 0.0

    for i in range(split, n_sub):
        if skipped[i]:
            left_mp[i] = left_mp[i - 1]
            continue

        value, is_exact = _backward_search(ts, m, i, threshold)
        left_mp[i] = value
        exact[i] = is_exact
        if is_exact and value > threshold:
            buffer_pos[buffer_len] = i
            buffer_val[buffer_len] = value
            buffer_len += 1
            if buffer_pos[buffer_len - 1] - buffer_pos[0] >= min_span:
                separated, _ = _greedy_separated(
                    buffer_pos[:buffer_len],
                    buffer_val[:buffer_len],
                    n_discords,
                    2 * radius,
                    blocked,
                )
                if len(separated) == n_discords and separated[-1] > threshold:
                    threshold = separated[-1]
                    kept = 0
                    for b in range(buffer_len):
                        if buffer_val[b] > threshold:
                            buffer_pos[kept] = buffer_pos[b]
                            buffer_val[kept] = buffer_val[b]
                            kept += 1
                    buffer_len = kept

        start = i + m
        if lookahead > 0 and start < n_sub and (threshold > 0.0 or n_discords == 1):
            end = min(start + lookahead, n)
            if end - start >= m:
                dist = _mass(ts[start:end], ts[i : i + m])
                for j in range(len(dist)):
                    if dist[j] <= threshold:
                        skipped[start + j] = True

    pruning_rate = np.sum(skipped[split:]) / (n_sub - split)

    candidates = np.flatnonzero(exact & (left_mp >= threshold))
    scores, locations = _greedy_separated(
        candidates, left_mp[candidates], n_discords, radius, blocked
    )
    return left_mp, scores, locations, pruning_rate
