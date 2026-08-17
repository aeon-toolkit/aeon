"""MADRID anomaly detector."""

__maintainer__ = ["JayeshSuryavanshi"]
__all__ = ["MADRID"]

import warnings

import numpy as np
from numba import njit

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.utils.numba.general import AEON_NUMBA_STD_THRESHOLD
from aeon.utils.numba.stats import std
from aeon.utils.windowing import reverse_windowing


class MADRID(BaseSeriesAnomalyDetector):
    """MADRID multi-length discord anomaly detector.

    MADRID is a discord discovery algorithm that finds time series anomalies of
    *all* lengths at once [1]_, rather than committing to a single subsequence
    length. Instead of committing to a single subsequence
    length, MADRID runs the left-discord matrix-profile method DAMP for every
    subsequence length in a candidate set and combines the length-normalised
    discord profiles into a single per-point anomaly score. It is the faster
    successor of the MERLIN discord discovery algorithm.

    For each candidate length ``m`` the algorithm computes an approximate left
    matrix profile with DAMP (the distance from each subsequence to its nearest
    neighbour that starts strictly to its left), normalises it by ``sqrt(m)`` so
    that profiles of different lengths are comparable, and stores it as a row of a
    multi-length discord table ``M``. Row scores of ``M`` describe subsequences
    *starting* at each position; :meth:`predict` converts them to pointwise
    scores, as the aeon contract requires: within each length, every point
    covered by a subsequence inherits that subsequence's score (reverse
    windowing with a max reduction), and the final score of a point is the
    maximum over lengths, i.e. the score of the most anomalous subsequence of
    any candidate length that covers it. Higher scores indicate more anomalous
    points.

    Points before ``train_test_split`` form a warm-up (training) region that is
    only used as reference history and is always scored zero, mirroring DAMP's
    left-discord definition.

    Parameters
    ----------
    min_length : int, default=8
        Minimum subsequence length in the candidate set. Must be at least 4.
    max_length : int, default=50
        Maximum subsequence length in the candidate set. Must be at most half the
        length of the series.
    step_size : int, default=1
        Step between consecutive candidate subsequence lengths. The candidate set
        is ``range(min_length, max_length + 1, step_size)``.
    train_test_split : int, float or None, default=None
        Location of the split point between the warm-up (training) region and the
        region searched for anomalies. An ``int`` is used directly as the split
        index. A ``float`` in ``(0, 1)`` is interpreted as a fraction of the series
        length. If ``None``, a warm-up of ``max(max_length, len(X) // 5)`` points
        is used. The split must satisfy ``max_length <= split <= len(X) -
        max_length``.

    References
    ----------
    .. [1] Yue Lu, Thirumalai Vinjamoor Akhil Srinivas, Takaaki Nakamura, Makoto
           Imamura and Eamonn Keogh, "Matrix Profile XXX: MADRID: A Hyper-Anytime
           and Parameter-Free Algorithm to Find Time Series Anomalies of All
           Lengths," 2023 IEEE International Conference on Data Mining (ICDM),
           Shanghai, China, 2023, pp. 1199-1204.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.distance_based import MADRID
    >>> rng = np.random.default_rng(2)
    >>> X = np.sin(np.linspace(0, 12 * np.pi, 120)) + rng.normal(0, 0.05, 120)
    >>> X[70:78] = 2.5  # inject an anomalous flat segment
    >>> detector = MADRID(min_length=6, max_length=12, train_test_split=24)
    >>> scores = detector.fit_predict(X)
    >>> bool(65 <= int(np.argmax(scores)) <= 78)
    True
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": False,
        "capability:missing_values": False,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
    }

    def __init__(
        self,
        min_length=8,
        max_length=50,
        step_size=1,
        train_test_split=None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.step_size = step_size
        self.train_test_split = train_test_split

        super().__init__(axis=1)

    def _predict(self, X):
        X = X.squeeze()
        n = X.shape[0]

        if self.step_size < 1:
            raise ValueError(f"step_size {self.step_size} must be at least 1")
        elif self.min_length < 4:
            raise ValueError("min_length must be at least 4")
        elif self.min_length > self.max_length:
            raise ValueError(
                f"min_length {self.min_length} must be less than or equal to "
                f"max_length {self.max_length}"
            )
        elif n < self.min_length:
            raise ValueError(
                f"Series length of X {n} is less than min_length {self.min_length}"
            )
        elif int(n / 2) < self.max_length:
            raise ValueError(
                f"Series length of X {n} must be at least double max_length "
                f"{self.max_length}"
            )

        split = self._resolve_split(n)
        if split < self.max_length or split > n - self.max_length:
            raise ValueError(
                f"train_test_split resolved to {split}, but it must lie in "
                f"[max_length, len(X) - max_length] = [{self.max_length}, "
                f"{n - self.max_length}]"
            )

        for i in range(n - self.min_length + 1):
            if std(X[i : i + self.min_length]) <= AEON_NUMBA_STD_THRESHOLD:
                warnings.warn(
                    "There is a region close to constant that will cause the "
                    "results to be unstable. It is suggested to delete the "
                    "constant region or try again with a longer min_length.",
                    stacklevel=2,
                )
                break

        m_set = np.arange(
            self.min_length, self.max_length + 1, self.step_size, dtype=np.int64
        )

        discord_table, _, _ = _madrid(
            np.ascontiguousarray(X, dtype=np.float64), split, m_set
        )

        # Convert subsequence scores to pointwise scores. Each row of the discord
        # table scores subsequences *starting* at each position; a point's
        # anomalousness is that of the subsequences covering it. Within a length,
        # every covered point inherits the window score via reverse windowing with
        # a max reduction (a discord score is a property of the whole subsequence,
        # and a mean would dilute a single-window discord with its normal
        # neighbours); across lengths the maximum is taken, so a point is as
        # anomalous as the most anomalous subsequence of any length covering it.
        point_scores = np.zeros(n)
        for i, m in enumerate(m_set):
            windowed = discord_table[i, : n - int(m) + 1]
            pointwise = reverse_windowing(windowed, int(m), np.nanmax)
            point_scores = np.maximum(point_scores, pointwise)
        return point_scores

    def _resolve_split(self, n):
        split = self.train_test_split
        if split is None:
            return max(self.max_length, n // 5)
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
        return {"min_length": 4, "max_length": 7, "train_test_split": 8}


@njit(cache=True, fastmath=True)
def _next_pow2(x):
    return int(np.ceil(np.log2(x)))


@njit(cache=True, fastmath=True)
def _moving_mean_std(x, m):
    n = len(x)
    result_mean = np.empty(n - m + 1, dtype=np.float64)
    result_std = np.empty(n - m + 1, dtype=np.float64)

    sum_x = 0.0
    sum_x_sq = 0.0
    for k in range(m):
        sum_x += x[k]
        sum_x_sq += x[k] * x[k]

    result_mean[0] = sum_x / m
    var0 = (sum_x_sq / m) - (result_mean[0] * result_mean[0])
    result_std[0] = np.sqrt(var0) if var0 > 0.0 else 0.0

    for i in range(1, n - m + 1):
        sum_x += x[i + m - 1] - x[i - 1]
        sum_x_sq += x[i + m - 1] * x[i + m - 1] - x[i - 1] * x[i - 1]
        mean = sum_x / m
        result_mean[i] = mean
        var = (sum_x_sq / m) - (mean * mean)
        result_std[i] = np.sqrt(var) if var > 0.0 else 0.0

    return result_mean, result_std


@njit(cache=True, fastmath=True)
def _sliding_dot_product(query, ts):
    m = len(query)
    n = len(ts)
    length = n - m + 1
    qt = np.empty(length, dtype=np.float64)
    for j in range(length):
        s = 0.0
        for k in range(m):
            s += query[k] * ts[j + k]
        qt[j] = s
    return qt


@njit(cache=True)
def _mass(ts, query):
    """Mueen's Algorithm for Similarity Search (z-normalised distance profile)."""
    m = len(query)
    length = len(ts) - m + 1

    mean_q = np.mean(query)
    std_q = np.std(query)
    if std_q < AEON_NUMBA_STD_THRESHOLD:
        std_q = AEON_NUMBA_STD_THRESHOLD

    mean_t, std_t = _moving_mean_std(ts, m)
    qt = _sliding_dot_product(query, ts)

    dist = np.empty(length, dtype=np.float64)
    for j in range(length):
        s = std_t[j]
        if s < AEON_NUMBA_STD_THRESHOLD:
            s = AEON_NUMBA_STD_THRESHOLD
        val = 2.0 * (m - (qt[j] - m * mean_t[j] * mean_q) / (s * std_q))
        dist[j] = np.sqrt(val) if val > 0.0 else 0.0
    return dist


@njit(cache=True)
def _damp_forward_processing(ts, m, i, best_so_far, pruned):
    n = len(ts)
    if i + m >= n - m + 1:
        return pruned
    lookahead = 2 ** _next_pow2(m)
    start = i + m
    end = min(start + lookahead, n)
    d_i = _mass(ts[start:end], ts[i : i + m])
    for k in range(len(d_i)):
        if d_i[k] <= best_so_far:
            pruned[start + k] = False
    return pruned


@njit(cache=True)
def _damp_backward_processing(ts, m, i, best_so_far):
    a_mp_i = np.inf
    a_mp_i_candidate = np.inf
    prefix = 2 ** _next_pow2(m)
    endpoint = i
    while a_mp_i_candidate >= best_so_far:
        if endpoint - prefix <= 0:
            a_mp_i = np.nanmin(_mass(ts[:endpoint], ts[i : i + m]))
            if a_mp_i < a_mp_i_candidate:
                a_mp_i_candidate = a_mp_i
            if a_mp_i_candidate > best_so_far and a_mp_i_candidate != np.inf:
                best_so_far = a_mp_i_candidate
            break
        else:
            a_mp_i = np.nanmin(_mass(ts[endpoint - prefix : endpoint], ts[i : i + m]))
            if a_mp_i < a_mp_i_candidate:
                a_mp_i_candidate = a_mp_i
            if a_mp_i < best_so_far:
                break
            else:
                endpoint = endpoint - prefix + m - 1
                prefix = 2 * prefix
    if a_mp_i_candidate == np.inf:
        a_mp_i_candidate = 0.0
    return a_mp_i_candidate, best_so_far


@njit(cache=True)
def _damp(ts, m, split, a_mp, best_so_far):
    """DAMP left-discord approximate matrix profile for one subsequence length."""
    n = len(ts)
    pruned = np.ones(n - m + 1, dtype=np.bool_)
    if best_so_far > 0.0:
        pruned = _damp_forward_processing(ts, m, split, best_so_far, pruned)

    for i in range(split, n - m + 1):
        if not pruned[i]:
            a_mp[i] = a_mp[i - 1]
        else:
            a_mp[i], best_so_far = _damp_backward_processing(ts, m, i, best_so_far)
            pruned = _damp_forward_processing(ts, m, i, best_so_far, pruned)

    return best_so_far, a_mp


@njit(cache=True)
def _madrid_warm_up(ts, split, discord_table, bsf, bsf_loc, m_set, done):
    n = len(ts)
    n_lengths = len(m_set)
    warmup_pointers = np.array([n_lengths // 2, 0, n_lengths - 1])
    for pointer in warmup_pointers:
        m_w = m_set[pointer]
        length = n - m_w + 1
        root = np.sqrt(m_w)
        a_mp_in = root * discord_table[pointer, :length].copy()
        discord_score, left_mp = _damp(ts, m_w, split, a_mp_in, root * bsf[pointer])
        bsf[pointer] = discord_score / root
        discord_table[pointer, :length] = left_mp / root
        loc = int(np.argmax(left_mp))
        bsf_loc[pointer] = loc
        done[pointer] = True

        for p in range(n_lengths):
            if done[p]:
                continue
            m = m_set[p]
            q_end = min(loc + m, n)
            q_len = q_end - loc
            if loc < m or q_len < 4:
                continue
            score = np.nanmin(_mass(ts[:loc], ts[loc:q_end])) / np.sqrt(m)
            discord_table[p, loc] = score
            if bsf[p] < score:
                bsf[p] = score
                bsf_loc[p] = loc
    return discord_table, bsf, bsf_loc, done


@njit(cache=True)
def _madrid_main(ts, split, discord_table, bsf, bsf_loc, m_set, done):
    n = len(ts)
    for p in range(len(m_set)):
        if done[p]:
            continue
        m = m_set[p]
        length = n - m + 1
        root = np.sqrt(m)
        a_mp_in = root * discord_table[p, :length].copy()
        discord_score, left_mp = _damp(ts, m, split, a_mp_in, root * bsf[p])
        discord_table[p, :length] = left_mp / root
        bsf_loc[p] = int(np.argmax(left_mp))
        bsf[p] = discord_score / root
        done[p] = True
    return discord_table, bsf, bsf_loc, done


@njit(cache=True)
def _madrid(ts, split, m_set):
    """Run MADRID and return the multi-length discord table and best-so-far info."""
    n = len(ts)
    n_lengths = len(m_set)
    discord_table = np.zeros((n_lengths, n))
    bsf = np.zeros(n_lengths)
    bsf_loc = np.zeros(n_lengths, dtype=np.int64)
    done = np.zeros(n_lengths, dtype=np.bool_)

    discord_table, bsf, bsf_loc, done = _madrid_warm_up(
        ts, split, discord_table, bsf, bsf_loc, m_set, done
    )
    discord_table, bsf, bsf_loc, done = _madrid_main(
        ts, split, discord_table, bsf, bsf_loc, m_set, done
    )
    return discord_table, bsf, bsf_loc
