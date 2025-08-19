"""AutoPlait Segmenter.

Python implementation based on original C code. Originally written by Yasuko Matsubara.
Translated to Python in 2025 with the aid of LLMs.
"""

import math
import random
import sys
import warnings

__all__ = ["AutoPlaitSegmenter"]

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from aeon.segmentation.base import BaseSegmenter

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Numerical thresholds
EPSILON = 1e-6
ONE_MINUS_EPSILON = 0.999999
VERY_LARGE_COST = 1e11
MAX_NORMALISATION_VALUE = 1.0

# Segment signs
S0 = 1
S1 = -1

# Math
PI = math.pi
E = math.e
FLOATING_POINT_CONSTANT = 4 * 8
LOG_EPSILON = np.log(EPSILON)

# Output options
ENABLE_HMM_PRINT = True
OUTPUT_VITERBI_PATH = False


class AutoPlaitSegmenter(BaseSegmenter):
    """AutoPlait Segmentation.

    AutoPlait [1]_ is a fully automatic segmentation algorithm for multivariate
    time series. AutoPlait models a time series using multiple
    Hidden Markov Models (HMM) and iteratively attempts to split
    HMMs into two smaller HMMs that have a lower Minimum Description Length (MDL)
    than the original larger HMM. AutoPlait is able to identify common patterns
    or 'regimes' in a time series (segments that model the same underlying state).
    The index points at which a time series switches from being model by optimal
    HMM_i to optimal HMM_j are cut points of the time series.

    Parameters
    ----------
    max_segments : int, default=250
        Maximum number of segments allowed during segmentation.
    max_sequence_length : int, default=10000000
        Maximum sequence length to consider; longer sequences are truncated.
    min_k : int, default=1
        Minimum number of hidden states to consider for HMM modeling.
    max_k : int, default=16
        Maximum number of hidden states to consider for HMM modeling.
    num_samples : int, default=10
        Number of random samples used during regime initialization.
    max_baumn_segments : int, default=3
        Maximum number of segments used during Baum-Welch parameter estimation.
    delta : int, default=1
        Delta regularization term controlling split preference.
    max_iter : int, default=10
        Maximum number of iterations allowed for regime splitting.
    max_k_means_iter : int, default=100
        Maximum number of iterations for KMeans clustering inside regime initialization.
    min_infer_iter : int, default=3
        Minimum number of iterations for regime inference during split.
    max_infer_iter : int, default=10
        Maximum number of iterations for regime inference during split.
    default_variance : float, default=10
        Default variance value for initializing emission probabilities.
    max_variance : float, default=inf
        Upper bound on allowed variance values during modeling.
    min_variance : float, optional (defaults to EPSILON)
        Lower bound on allowed variance values during modeling.
    segment_sample_ratio : float, default=0.03
        Ratio of segments to sample when pruning noisy splits.
    regime_sample_ratio : float, default=0.03
        Ratio controlling acceptance of split regimes based on MDL cost.
    sampling_lm : float, default=0.1
        Proportion of sequence length to use during sampling for KMeans initialization.
    seed : int, optional
        Random seed used for reproducibility.
    normalise : bool, default=True
        Whether to apply Z-normalization to input sequences.
    verbose : bool, default=False
        If True, print progress information during segmentation.


    References
    ----------
    .. [1] Matsubara, Yasuko and Sakurai, Yasushi and Faloutsos, Christos.
    "AutoPlait: Automatic Mining of Co-evolving Time Sequences.", SIGMOD, 2014.


    Examples
    --------
    >>> from aeon.segmentation import AutoPlaitSegmenter
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, _, cps = load_gun_point_segmentation()
    >>> X = np.array(X)
    >>> autoplait = AutoPlaitSegmenter()
    >>> found_cps = autoplait.fit_predict(X, axis=0)
    >>> regimes = autoplait.get_regime_labels()
    """

    _tags = {
        "returns_dense": True,
        "capability:univariate": False,
        "capability:multivariate": True,
    }

    def __init__(
        self,
        max_segments: int = 250,
        max_sequence_length: int = 10_000_000,
        min_k: int = 1,
        max_k: int = 16,
        num_samples: int = 10,
        max_baumn_segments: int = 3,
        delta: int = 1,
        max_iter: int = 10,
        max_k_means_iter: int = 100,
        min_infer_iter: int = 3,
        max_infer_iter: int = 10,
        default_variance: float = 10,
        max_variance: float = float("inf"),
        min_variance: float = 1e-6,
        segment_sample_ratio: float = 0.03,
        regime_sample_ratio: float = 0.03,
        sampling_lm: float = 0.1,
        seed: int | None = None,
        normalise: bool = True,
        verbose: bool = False,
    ):

        self.max_segments = max_segments
        self.max_sequence_length = max_sequence_length
        self.min_k = min_k
        self.max_k = max_k
        self.num_samples = num_samples
        self.max_baumn_segments = max_baumn_segments
        self.delta = delta
        self.max_iter = max_iter
        self.max_k_means_iter = max_k_means_iter
        self.min_infer_iter = min_infer_iter
        self.max_infer_iter = max_infer_iter
        self.default_variance = default_variance
        self.max_variance = max_variance
        self.min_variance = min_variance
        self.segment_sample_ratio = segment_sample_ratio
        self.regime_sample_ratio = regime_sample_ratio
        self.sampling_lm = sampling_lm

        self.x = None
        self.maxc = 0
        self.maxseg = 0
        self.d = 0
        self.lmax = max_sequence_length
        self.cps = _CPS(self)
        self.baum = _BaumWelch(self)
        self.q = None
        self.vit = _Viterbi(self)
        self.vit2 = _Viterbi(self)
        self.x_tmp = None
        self.s = None
        self.C = []
        self.Opt = []
        self.S = []
        self.costT = 0.0
        self.U = _Regime(self)

        self.verbose = verbose
        self.axis = 0
        self.seed = seed
        self.normalise = normalise
        random.seed(seed)

        super().__init__(axis=0)

    def _set_up(self):
        """
        Create internal AutoPlait structures.

        Creates data structures for Viterbi decoding, Baum-Welch training,
        and regime segmentation boxes based on configured parameters.

        This must be called after input data dimensions are known.
        """
        self.maxc = int(math.log(self.lmax) + 20)
        self.maxseg = self.max_segments  # int(ws.lmax * 0.1)

        # Allocate Viterbi
        self.vit._set_up(self.lmax, self.max_k)
        self.vit2._set_up(self.lmax, self.max_k)
        self.q = np.zeros(self.lmax, dtype=np.int32)  # ivector(0, ws.lmax)
        self.cps._set_up(self.max_k, self.lmax)

        # Allocate Baum
        n = self.maxseg
        if n > self.max_baumn_segments:
            n = self.max_baumn_segments

        self.x_tmp = [_Input() for _ in range(n)]
        self.baum._set_up(n, self.lmax, self.max_k, self.d)

        # Allocate segbox
        self.s = [_Regime(self) for _ in range(self.maxc)]
        for i in range(self.maxc):
            self.s[i]._set_up()

        for i in range(self.maxc):
            self.S.append(self.s[i])

        # For uniform sampling
        self.U._set_up()

    def _reshape_and_normalise(self, X):
        """
        Fit the AutoPlait segmenter to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input time series array, either univariate (1D) or multivariate (2D).
        y : Ignored
            Exists for compatibility with scikit-learn-style estimators.

        Returns
        -------
        self : AutoPlaitSegmenter
            Fitted segmenter instance.
        """
        if X.ndim == 1:  # Reshape univariate data from (n,) to (n,1)
            X = X.reshape(-1, 1)

        self.d = X.shape[1]
        inp = _Input()
        inp.length = min(X.shape[0], self.lmax)
        inp.id = 0
        inp.observations = X
        self.x = inp
        self.lmax = self.x.length
        self._set_up()
        if self.normalise:
            self._normalise()

    def _predict(self, X):
        """
        Predict change points on the fitted input sequence.

        Runs the full AutoPlait segmentation pipeline
        and extracts the list of detected change points.

        Parameters
        ----------
        X : np.ndarray
            Ignored (exists for compatibility). Uses training data X passed to `fit`.

        Returns
        -------
        np.ndarray
            Array of detected change points (indices).
        """
        self._reshape_and_normalise(X)
        if self.verbose:
            sys.stdout.write("---------\nr|m|Cost \n---------\n")  # noqa: T201

        self._plait()
        output = []
        for i in range(len(self.Opt)):
            regime = self.Opt[i]
            for j in range(regime.num_segments):
                output.append(
                    (
                        (
                            regime.segments[j]["start"]
                            + regime.segments[j]["duration"]
                            - 1
                        ),
                        i,
                    )
                )
        output = sorted(output, key=lambda x: x[0])
        cps = [x[0] for x in output][:-1]
        self.regimes = np.array([x[1] for x in output])
        return np.array(cps)

    def _get_new_regime(self, parent_label="", child_label=""):
        """
        Create a new, initialized Regime object from the reusable segment pool.

        Parameters
        ----------
        parent_label : str, default=''
            Label of the parent regime (used for hierarchical labeling).
        child_label : str, default=''
            Label suffix for the new regime.

        Returns
        -------
        _Regime
            A new regime ready for modeling.
        """
        s = self.S.pop()
        if s is None:
            raise ValueError("Too small maxc")

        s.reset()
        s.model.reset(self.max_k, self.d)
        s.label = f"{parent_label}{child_label}"
        s.delta = 1.0 / float(self.lmax)

        return s

    def get_regime_labels(self):
        """
        Return the list of regime labels after segmentation.

        Returns
        -------
        np.ndarray
            Regime index labels assigned to each final segment.
        """
        self._check_is_fitted()
        return self.regimes

    def _plait(self):
        """
        Run the AutoPlait segmentation algorithm.

        Iteratively splits regimes by optimizing Minimum Description Length (MDL) cost.
        Builds a segmentation hierarchy until no further cost reduction is achieved.
        """
        current_regime = self._get_new_regime("", "")
        current_regime.costT = VERY_LARGE_COST
        current_regime.add_segment(0, self.x.length)
        current_regime.estimate_parameters()
        self.C.append(current_regime)

        while True:
            self.costT = self._compute_total_mdl()
            if not self.C:
                break
            current_regime = self.C.pop()

            # Create new segment sets
            regime0 = self._get_new_regime(current_regime.label, "0")
            regime1 = self._get_new_regime(current_regime.label, "1")

            # Try to split regime: current_regime->(regime0,regime1)
            self.cps._regime_split(current_regime, regime0, regime1)
            costT_regime01 = regime0.costT + regime1.costT

            # Split or not
            if (
                costT_regime01 + current_regime.costT * self.regime_sample_ratio
                < current_regime.costT
            ):
                self.C.append(regime0)
                self.C.append(regime1)
                self.S.append(current_regime)
            else:
                self.Opt.append(current_regime)
                self.S.append(regime0)
                self.S.append(regime1)

    def _normalise(self):
        """
        Apply Z-normalization (mean=0, std=1) independently to each feature dimension.

        Modifies the input sequence in-place.
        """
        for d in range(self.d):
            mean = 0.0
            std = 0.0
            cnt = 0

            x = self.x
            cnt += x.length
            for j in range(x.length):
                mean += x.observations[j][d]

            mean /= cnt

            # Compute std
            x = self.x
            for j in range(x.length):
                std += (x.observations[j][d] - mean) ** 2

            std = math.sqrt(std / cnt)

            # Normalize
            x = self.x
            for j in range(x.length):
                x.observations[j][d] = (
                    MAX_NORMALISATION_VALUE * (x.observations[j][d] - mean) / std
                )

    def _compute_total_mdl(self):
        """
        Compute the total Minimum Description Length (MDL).

        For the current segmentation state.

        Returns
        -------
        float
            Total MDL cost in bits.
        """
        r = len(self.Opt) + len(self.C)
        m = sum(r.num_segments for r in self.Opt) + sum(r.num_segments for r in self.C)
        cost = sum(r.costT for r in self.Opt) + sum(r.costT for r in self.C)
        costT = (
            cost + log_s(r) + log_s(m) + m * log_2(r) + (4 * 8) * r * r
        )  # FLOATING_POINT_CONSTANT*r*r

        if self.verbose:
            sys.stdout.write(f"{r} {m} {costT:.0f} \n")  # noqa: T201

        return costT


@dataclass
class _Input:
    """
    Input sequence container class.

    Used to represent a (sub)sequence extracted from a full time series
    during segmentation and HMM modeling.

    Attributes
    ----------
    id : int
        Unique identifier for the input sequence.
    tag : str
        Tag name associated with the sequence.
    parent : int
        ID of the parent sequence (if split from another sequence).
    start_idx : int
        Start index of the subsequence within the full sequence.
    observations : np.ndarray
        Observation sequence (2D array: [time steps, features]).
    length : int
        Length of the subsequence (number of time steps).
    pid : int
        Process ID or another label (depending on usage).
    """

    id: int = 0
    tag: str = ""
    parent: int = 0
    start_idx: int = 0
    observations: np.ndarray = None
    length: int = 0
    pid: int = 0


class _Regime:
    """
    A structure representing a segmented regime of a time series.

    Attributes
    ----------
        segments (np.ndarray): Structured NumPy array of segments,
        each with 'start' and 'duration'.
        nSb (int): Number of active segments.
        total_length (int): Total combined duration of all segments.
        costT (float): Total cost including model and encoding.
        costC (float): Cost of encoding the data.
        optimal (bool): Whether this regime is considered optimal.
        label (str): Label for the regime.
        model (_HMM): Hidden Markov Model associated with this regime.
        delta (float): Proportion of time spent in this regime relative
        to total sequence length.
    """

    def __init__(self, ap_seg, label=""):

        self.ap_seg = ap_seg

        segment_dtype = np.dtype([("start", np.int32), ("duration", np.int32)])
        self.segments = np.zeros(self.ap_seg.max_segments, dtype=segment_dtype)
        self.num_segments = 0
        self.total_length = 0
        self.costT = 0.0
        self.costC = 0.0
        self.optimal = False
        self.label = label
        self.model = _HMM(self.ap_seg.default_variance)
        self.delta = 0.0

    def _set_up(self):
        """
        Initialize or reset the internal model associated with the regime.

        Resets segment count, sets up the associated HMM model structure and
        marks regime as non-optimal.
        """
        self.num_segments = 0
        self.optimal = False
        self.model._set_up(self.ap_seg.max_k, self.ap_seg.d)

    def reset(self) -> None:
        """
        Reset the regime to an empty state.

        Clears all segments and resets costs.
        """
        self.num_segments = 0
        self.total_length = 0
        self.costC = float("inf")
        self.costT = float("inf")

    def add_segment(self, start: int, duration: int) -> None:
        """
        Add a new non-overlapping segment to the regime.

        If overlaps are detected after insertion, they are automatically merged.

        Parameters
        ----------
        start : int
            Start index of the segment.
        duration : int
            Length of the segment.

        Raises
        ------
        RuntimeError
            If the number of segments exceeds the maximum allowed (`max_segments`).
        """
        if duration <= 0:
            return
        start = max(0, start)

        # Find insertion point
        loc = 0
        while loc < self.num_segments and self.segments[loc]["start"] <= start:
            loc += 1

        # Shift elements to make space
        if self.num_segments >= self.ap_seg.max_segments:
            raise RuntimeError("Too many segments")

        if loc < self.num_segments:
            self.segments[loc + 1 : self.num_segments + 1] = self.segments[
                loc : self.num_segments
            ]

        # Insert new segment
        self.segments[loc]["start"] = start
        self.segments[loc]["duration"] = duration
        self.num_segments += 1

        self._remove_overlap()
        self.total_length = self.segments[: self.num_segments]["duration"].sum()

    def _remove_overlap(self) -> None:
        """
        Merge overlapping or adjacent segments.

        Ensures all segments are non-overlapping and sorted by start time.
        """
        i = 0
        while i < self.num_segments - 1:
            starts = self.segments[: self.num_segments]["start"]
            durations = self.segments[: self.num_segments]["duration"]
            ends = starts + durations

            seg0_start = starts[i]
            seg0_end = ends[i]
            seg1_start = starts[i + 1]
            seg1_end = ends[i + 1]

            if seg0_end + 1 >= seg1_start:
                # Merge seg0 and seg1
                self.segments[i]["duration"] = max(seg0_end, seg1_end) - seg0_start
                self.remove_segment(i + 1)
                # Don't advance i
            else:
                i += 1

    def remove_segment(self, index: int) -> None:
        """
        Remove the segment at the specified index.

        Parameters
        ----------
        index : int
            Index of the segment to remove.

        Raises
        ------
        IndexError
            If the specified index is out of bounds.
        """
        if index >= self.num_segments:
            raise IndexError(
                f"remove_segment: index {index} out of bounds (nSb={self.num_segments})"
            )

        self.total_length -= self.segments[index]["duration"]

        if index < self.num_segments - 1:
            self.segments[index : self.num_segments - 1] = self.segments[
                index + 1 : self.num_segments
            ]

        self.segments[self.num_segments - 1] = (0, 0)  # Optional clear
        self.num_segments -= 1

    def add_segment_with_overlap(self, start: int, duration: int) -> None:
        """
        Add a segment to the regime without merging overlaps.

        Parameters
        ----------
        start : int
            Start index of the segment.
        duration : int
            Length of the segment.

        Raises
        ------
        RuntimeError
            If the number of segments exceeds the maximum allowed (`max_segments`).
        """
        if duration <= 0:
            return
        if self.num_segments >= self.ap_seg.max_segments:
            raise RuntimeError("Exceeded maximum number of segments")

        self.segments[self.num_segments]["start"] = start
        self.segments[self.num_segments]["duration"] = duration
        self.num_segments += 1
        self.total_length += duration

    def copy_to(self, target: "_Regime") -> None:
        """
        Copy this regime's segments, costs, and attributes into another regime.

        Parameters
        ----------
        target : _Regime
            The regime to copy data into.
        """
        # Copy active segment data using slicing
        target.segments[: self.num_segments] = self.segments[: self.num_segments]

        target.num_segments = self.num_segments
        target.total_length = self.total_length
        target.costT = self.costT
        target.costC = self.costC
        target.delta = self.delta
        target.label = self.label

    def max_segment_index(self) -> int:
        """
        Return the index of the segment with the maximum duration.

        Returns
        -------
        int
            Index of the largest segment, or -1 if no segments exist.
        """
        if self.num_segments == 0:
            return -1
        return int(np.argmax(self.segments[: self.num_segments]["duration"]))

    def estimate_parameters(self) -> None:
        """
        Estimate the optimal HMM parameters for this regime.

        Searches for the best number of states by minimizing the MDL cost.
        """
        self.costT = float("inf")
        optk = self.ap_seg.min_k

        for k in range(self.ap_seg.min_k, self.ap_seg.max_k + 1):
            previous_cost = self.costT
            self.estimate_hmm_k(k)
            self.compute_likelihood_mdl()

            if self.costT > previous_cost:
                optk = max(self.ap_seg.min_k, k - 1)
                break

        optk = min(self.ap_seg.max_k, max(self.ap_seg.min_k, optk))

        self.estimate_hmm_k(optk)
        self.compute_likelihood_mdl()

    def estimate_hmm_k(self, k: int) -> None:
        """
        Estimate the HMM parameters for a fixed number of states.

        Parameters
        ----------
        k : int
            Number of hidden states to fit.
        """
        k = min(
            self.ap_seg.max_k, max(self.ap_seg.min_k, k)
        )  # Clamp k to [MIN_K, MAX_K]

        num_segments = min(self.num_segments, self.ap_seg.max_baumn_segments)
        for i in range(num_segments):
            _split_sequence(
                self.ap_seg.x,
                self.segments[i]["start"],
                self.segments[i]["duration"],
                self.ap_seg.x_tmp[i],
            )

        self.model.k = k
        self.model.n = 0
        self.model.reset(k, self.ap_seg.d)

        self.ap_seg.baum.baum_welch(
            self.model, num_segments, self.ap_seg.x_tmp, use_k_means=True
        )
        self.delta = self.num_segments / self.total_length if self.total_length else 0.0

    def mdl(self) -> float:
        """
        Compute the Minimum Description Length (MDL) of the regime.

        Returns
        -------
        float
            Total MDL cost (encoding + model complexity).
        """
        k = self.model.k
        d = self.model.d
        m = self.total_length

        durations = self.segments[: self.num_segments]["duration"]
        costLen = np.log2(durations).sum()
        costLen += m * np.log2(k)

        costM = FLOATING_POINT_CONSTANT * (k + k * k + 2 * k * d) + log_s(k)
        return self.costC + costLen + costM

    def compute_likelihood_mdl(self) -> None:
        """
        Compute the likelihood cost and MDL cost for the regime.

        Uses Viterbi decoding to calculate log-likelihood over all segments.
        """
        if self.num_segments == 0:
            self.costC = float("inf")
            self.costT = float("inf")
            return

        starts = self.segments[: self.num_segments]["start"]
        durations = self.segments[: self.num_segments]["duration"]

        costs = np.fromiter(
            (
                self.ap_seg.vit._viterbi(self.model, self.delta, int(st), int(dur))
                for st, dur in zip(starts, durations)
            ),
            dtype=float,
            count=self.num_segments,
        )
        self.costC = costs.sum()
        self.costT = self.mdl()

    def _retain_largest_segment(self) -> None:
        """
        Retain only the single largest segment in the regime.

        All other segments are discarded.
        """
        index = self.max_segment_index()
        if index == -1:
            return

        self.reset()
        self.add_segment(
            self.segments[index]["start"], self.segments[index]["duration"]
        )


class _CPS:
    """
    CutPointSearch structure.

    Attributes
    ----------
        Pu, Pv, Pi, Pj: Probability arrays
        Su, Sv, Si, Sj: State arrays
        nSu, nSv, nSi, nSj: Counts
    """

    def __init__(self, ap_seg):
        self.ap_seg = ap_seg

        self.Pu = None
        self.Pv = None
        self.Pi = None
        self.Pj = None
        self.Su = None
        self.Sv = None
        self.Si = None
        self.Sj = None
        self.nSu = None
        self.nSv = None
        self.nSi = None
        self.nSj = None

    def _set_up(self, max_states, max_length):
        """
        Initialize internal arrays for Cut Point Search (CPS).

        Parameters
        ----------
        max_states : int
            Maximum number of HMM states to handle.
        max_length : int
            Maximum sequence length.
        """
        self.Pu = np.zeros(max_states, dtype=np.float64)
        self.Pv = np.zeros(max_states, dtype=np.float64)
        self.Pi = np.zeros(max_states, dtype=np.float64)
        self.Pj = np.zeros(max_states, dtype=np.float64)
        self.Su = np.zeros((max_states, max_length), dtype=np.int32)
        self.Sv = np.zeros((max_states, max_length), dtype=np.int32)
        self.Si = np.zeros((max_states, max_length), dtype=np.int32)
        self.Sj = np.zeros((max_states, max_length), dtype=np.int32)
        self.nSu = np.zeros(max_states, dtype=np.int32)
        self.nSv = np.zeros(max_states, dtype=np.int32)
        self.nSi = np.zeros(max_states, dtype=np.int32)
        self.nSj = np.zeros(max_states, dtype=np.int32)

    def _search_aux(self, start_idx, length, regime0, regime1):
        """
        Perform auxiliary dynamic programming search between two regimes.

        This tries to split a sequence starting at a given position
        into two candidate regimes by evaluating the likelihood of
        switching between the two HMMs.

        Parameters
        ----------
        start_idx : int
            Start index of the subsequence.
        length : int
            Length of the subsequence.
        regime0 : _Regime
            First candidate regime model.
        regime1 : _Regime
            Second candidate regime model.

        Returns
        -------
        float
            Negative log-likelihood (coding cost) of the best path.
        """
        # Extract HMMs and deltas
        m0, d0 = regime0.model, regime0.delta
        m1, d1 = regime1.model, regime1.delta
        k0, k1 = m0.k, m1.k
        observations = self.ap_seg.x.observations
        observation_window = observations[start_idx : start_idx + length]

        # Handle degenerate case early
        if d0 <= 0 or d1 <= 0:
            raise ValueError("Degenerate dlta <= 0")

        # Alias CPS structures
        Pu, Pv = self.Pu, self.Pv
        Pi, Pj = self.Pi, self.Pj
        Su, Sv = self.Su, self.Sv
        Si, Sj = self.Si, self.Sj
        nSu, nSv = self.nSu, self.nSv
        nSi, nSj = self.nSi, self.nSj

        # Reset path state
        nSu[:k0] = 0
        nSv[:k0] = 0
        nSi[:k1] = 0
        nSj[:k1] = 0

        # Precompute emissions
        log_emission0 = batch_log_pdf(
            m0, k0, length, observation_window
        )  # shape: (k0, length)
        log_emission1 = batch_log_pdf(
            m1, k1, length, observation_window
        )  # shape: (k1, length)

        # Precompute log(pi) and log(A) values
        log_pi0 = np.log(m0.pi + EPSILON)
        log_pi1 = np.log(m1.pi + EPSILON)
        log_A0 = np.log(m0.A + EPSILON)
        log_A1 = np.log(m1.A + EPSILON)

        # Precompute transition constants
        log_d0 = math.log(d0)
        log_d1 = math.log(d1)
        log_1md0 = math.log(1.0 - d0)
        log_1md1 = math.log(1.0 - d1)

        # Initialize log-probabilities at t = 0
        Pv[:k0] = log_d1 + log_pi0[:k0] + log_emission0[:, 0]
        Pj[:k1] = log_d0 + log_pi1[:k1] + log_emission1[:, 0]

        # Dynamic programming: for t >= 1
        for t in range(start_idx + 1, start_idx + length):
            offset = t - start_idx

            # --- Update Pu[t] ---

            # Find best path from previous Pj
            maxj = Pj[:k1].argmax()

            for u in range(k0):
                log_emit = log_emission0[u, offset]

                # Case: switch to m0
                switch_score = Pj[maxj] + log_d1 + log_pi0[u] + log_emit

                # Case: stay in m0 (vectorized over v)
                vals = log_1md0 + Pv[:k0] + log_A0[:k0, u] + log_emit
                maxv = int(np.argmax(vals))
                stay_score = vals[maxv]

                # Choose better path
                if switch_score > stay_score:
                    Pu[u] = switch_score
                    Su[u, : nSj[maxj]] = Sj[maxj, : nSj[maxj]]
                    nSu[u] = nSj[maxj]
                    Su[u][nSu[u]] = t
                    nSu[u] += 1
                else:
                    Pu[u] = stay_score
                    Su[u, : nSv[maxv]] = Sv[maxv, : nSv[maxv]]
                    nSu[u] = nSv[maxv]

            # --- Update Pi[t] ---

            maxv = Pv[:k0].argmax()

            for i in range(k1):
                log_emit = log_emission1[i, offset]

                # Case: switch to m1
                switch_score = Pv[maxv] + log_d0 + log_pi1[i] + log_emit

                # Case: stay in m1 (vectorized over j)
                vals = log_1md1 + Pj[:k1] + log_A1[:k1, i] + log_emit
                maxj = int(np.argmax(vals))
                stay_score = vals[maxj]

                if switch_score > stay_score:
                    Pi[i] = switch_score
                    Si[i, : nSv[maxv]] = Sv[maxv, : nSv[maxv]]
                    nSi[i] = nSv[maxv]
                    Si[i][nSi[i]] = t
                    nSi[i] += 1
                else:
                    Pi[i] = stay_score
                    Si[i, : nSj[maxj]] = Sj[maxj, : nSj[maxj]]
                    nSi[i] = nSj[maxj]

            # --- Swap buffers for next timestep ---
            Pu, Pv = Pv, Pu
            Pi, Pj = Pj, Pi
            Su, Sv = Sv, Su
            Si, Sj = Sj, Si
            nSu, nSv = nSv, nSu
            nSi, nSj = nSj, nSi

        # Determine which path has higher likelihood
        maxv = int(np.argmax(Pv[:k0]))
        maxj = int(np.argmax(Pj[:k1]))
        score_v = Pv[maxv]
        score_j = Pj[maxj]

        if score_v > score_j:
            path, npath, lh = Sv[maxv], nSv[maxv], score_v
            firstID = (-1) ** npath * S0
        else:
            path, npath, lh = Sj[maxj], nSj[maxj], score_j
            firstID = (-1) ** npath * S1

        # Segment reconstruction
        curSt = start_idx
        flip = firstID  # Either S0 or S1 depending on best regime

        for i in range(npath):
            nxtSt = path[i]
            target_regime = regime0 if flip == S0 else regime1
            target_regime.add_segment(curSt, nxtSt - curSt)
            curSt = nxtSt
            flip *= -1  # Alternate regime

        # Final segment
        target_regime = regime0 if flip == S0 else regime1
        target_regime.add_segment(curSt, start_idx + length - curSt)

        # Compute and return coding cost in bits
        costC = -lh / math.log(2.0)
        return costC

    def cut_point_search(self, current_regime, regime0, regime1):
        """
        Run a full Cut Point Search over all segments in the current regime.

        Parameters
        ----------
        current_regime : _Regime
            Source regime to split.
        regime0 : _Regime
            First candidate output regime.
        regime1 : _Regime
            Second candidate output regime.

        Returns
        -------
        float
            Total coding cost after the split.
        """
        regime0.reset()
        regime1.reset()

        lh = 0
        for i in range(current_regime.num_segments):
            lh += self._search_aux(
                current_regime.segments[i]["start"],
                current_regime.segments[i]["duration"],
                regime0,
                regime1,
            )

        return lh

    def _cut_point_search_with_noise_removal(
        self, current_regime, regime0, regime1, remove_noise
    ):
        """
        Perform Cut Point Search with optional noise removal.

        Parameters
        ----------
        current_regime : _Regime
            Source regime to split.
        regime0 : _Regime
            First output regime after splitting.
        regime1 : _Regime
            Second output regime after splitting.
        remove_noise : bool
            Whether to apply noise removal (True) or not (False).
        """
        self.cut_point_search(current_regime, regime0, regime1)

        if remove_noise:
            self._remove_noise(current_regime, regime0, regime1)

        regime0.compute_likelihood_mdl()
        regime1.compute_likelihood_mdl()

    def _estimate_regimes_aux(self, current_regime, regime0, regime1):
        """
        Estimate regime parameters by iterative refinement.

        Alternates between parameter estimation and cut point search
        to converge toward a better segmentation.

        Parameters
        ----------
        current_regime : _Regime
            The regime being split.
        regime0 : _Regime
            First candidate output regime.
        regime1 : _Regime
            Second candidate output regime.
        """
        opt0 = self.ap_seg._get_new_regime("", "")
        opt1 = self.ap_seg._get_new_regime("", "")

        for i in range(self.ap_seg.max_infer_iter):
            # Phase 1: Estimate parameters
            regime0._retain_largest_segment()
            regime1._retain_largest_segment()
            regime0.estimate_parameters()
            regime1.estimate_parameters()

            # Phase 2: Find cut-points
            self._cut_point_search_with_noise_removal(
                current_regime, regime0, regime1, True
            )

            if regime0.num_segments == 0 or regime1.num_segments == 0:
                break  # Avoid null inference

            # If improving, update the optimal segment set
            diff = (opt0.costT + opt1.costT) - (regime0.costT + regime1.costT)
            if diff > 0:
                regime0.copy_to(opt0)
                regime1.copy_to(opt1)
            # If not improving, then break iteration (efficient convergence)
            elif i >= self.ap_seg.min_infer_iter:
                break

        opt0.copy_to(regime0)
        opt1.copy_to(regime1)
        self.ap_seg.S.append(opt0)
        self.ap_seg.S.append(opt1)

    def _regime_split(self, current_regime, regime0, regime1):
        """
        Split a regime into two child regimes.

        Estimates the best centroids and refines using Cut Point Search.

        Parameters
        ----------
        current_regime : _Regime
            The regime to split.
        regime0 : _Regime
            First output regime after splitting.
        regime1 : _Regime
            Second output regime after splitting.
        """
        seedlen = int(math.ceil(self.ap_seg.lmax * self.ap_seg.sampling_lm))
        # Initialize HMM parameters
        self._find_centroid(
            current_regime, regime0, regime1, self.ap_seg.num_samples, seedlen
        )
        self._estimate_regimes_aux(current_regime, regime0, regime1)

        if regime0.num_segments == 0 or regime1.num_segments == 0:
            return

        # Final model estimation
        regime0.estimate_parameters()
        regime1.estimate_parameters()

    def _remove_noise(self, current_regime, regime0, regime1):
        """
        Attempt to remove noisy segments during Cut Point Search.

        Parameters
        ----------
        current_regime : _Regime
            Original unsplit regime.
        regime0 : _Regime
            First candidate output regime.
        regime1 : _Regime
            Second candidate output regime.
        """
        if regime0.num_segments <= 1 and regime1.num_segments <= 1:
            return

        # Default pruning
        sample_ratio = self.ap_seg.segment_sample_ratio
        self._remove_noise_aux(regime0, regime1, sample_ratio)
        costC = self._scan_min_diff(current_regime, regime0, regime1)

        # Optimal segment set
        opt0 = self.ap_seg._get_new_regime("", "")
        opt1 = self.ap_seg._get_new_regime("", "")
        regime0.copy_to(opt0)
        regime1.copy_to(opt1)
        prev = VERY_LARGE_COST

        # Find optimal pruning point
        while sample_ratio <= self.ap_seg.segment_sample_ratio * 10:
            if costC >= VERY_LARGE_COST:
                break

            sample_ratio *= 2
            self._remove_noise_aux(regime0, regime1, sample_ratio)

            if regime0.num_segments <= 1 or regime1.num_segments <= 1:
                break

            costC = self._scan_min_diff(current_regime, regime0, regime1)

            if prev > costC:
                regime0.copy_to(opt0)
                regime1.copy_to(opt1)
            else:
                break

            prev = costC

        opt0.copy_to(regime0)
        opt1.copy_to(regime1)
        self.ap_seg.S.append(opt0)
        self.ap_seg.S.append(opt1)

    def _find_centroid(
        self, current_regime, regime0, regime1, num_samples, seed_length
    ):
        """
        Find good initial seeds (centroids) for segmentation.

        Samples subsequences and selects the pair that minimizes combined cost.

        Parameters
        ----------
        current_regime : _Regime
            Regime being split.
        regime0 : _Regime
            First candidate output regime.
        regime1 : _Regime
            Second candidate output regime.
        num_samples : int
            Number of samples to evaluate.
        seed_length : int
            Length of the seed subsequences.

        Returns
        -------
        float
            Minimum combined cost of selected centroids.
        """
        costMin = VERY_LARGE_COST

        # Keep best seeds
        regime0stB, regime1stB, regime0lenB, regime1lenB = 0, 0, 0, 0  # Best

        # Make sample set
        uniform_set(current_regime, seed_length, num_samples, self.ap_seg.U)

        # Start uniform sampling
        for iter1 in range(self.ap_seg.U.num_segments):
            for iter2 in range(iter1 + 1, self.ap_seg.U.num_segments):
                uniform_sampling(
                    regime0, regime1, seed_length, iter1, iter2, self.ap_seg.U
                )

                if regime0.num_segments == 0 or regime1.num_segments == 0:
                    continue  # Not sufficient

                # Copy positions
                regime0stC = regime0.segments[0]["start"]
                regime0lenC = regime0.segments[0]["duration"]
                regime1stC = regime1.segments[0]["start"]
                regime1lenC = regime1.segments[0]["duration"]

                # Estimate HMM
                regime0.estimate_hmm_k(self.ap_seg.min_k)
                regime1.estimate_hmm_k(self.ap_seg.min_k)

                # Cut point search
                self._cut_point_search_with_noise_removal(
                    current_regime, regime0, regime1, True
                )

                if regime0.num_segments == 0 or regime1.num_segments == 0:
                    continue

                if costMin > regime0.costT + regime1.costT:
                    # Update best seeds
                    costMin = regime0.costT + regime1.costT
                    regime0stB = regime0stC
                    regime0lenB = regime0lenC
                    regime1stB = regime1stC
                    regime1lenB = regime1lenC

        if costMin == VERY_LARGE_COST:
            fixed_sampling(current_regime, regime0, regime1, seed_length)
            return VERY_LARGE_COST

        regime0.reset()
        regime1.reset()
        regime0.add_segment(regime0stB, regime0lenB)
        regime1.add_segment(regime1stB, regime1lenB)

        return costMin

    def _scan_min_diff(self, current_regime, regime0, regime1):
        """
        Scan regimes for the minimal coding cost improvement possible.

        Parameters
        ----------
        current_regime : _Regime
            Original regime.
        regime0 : _Regime
            First candidate regime.
        regime1 : _Regime
            Second candidate regime.

        Returns
        -------
        float
            Minimum difference cost found.
        """
        diff = [0.0]  # Use list to simulate pass-by-reference
        loc0 = self._find_min_diff(regime0, regime1, diff)
        loc1 = self._find_min_diff(regime1, regime0, diff)

        if loc0 == -1 or loc1 == -1:
            return VERY_LARGE_COST

        tmp0 = self.ap_seg._get_new_regime("", "")
        tmp1 = self.ap_seg._get_new_regime("", "")
        tmp0.add_segment(
            regime0.segments[loc0]["start"], regime0.segments[loc0]["duration"]
        )
        tmp1.add_segment(
            regime1.segments[loc1]["start"], regime1.segments[loc1]["duration"]
        )
        tmp0.estimate_hmm_k(self.ap_seg.min_k)
        tmp1.estimate_hmm_k(self.ap_seg.min_k)
        costC = self.cut_point_search(current_regime, tmp0, tmp1)
        self.ap_seg.S.append(tmp0)
        self.ap_seg.S.append(tmp1)

        return costC

    def _find_min_diff(self, regime0, regime1, min_diff_holder):
        """
        Find the segment in regime0 that has the minimum cost difference.

        Evaluates regime0 under regimes1's model.

        Parameters
        ----------
        regime0 : _Regime
            First regime.
        regime1 : _Regime
            Second regime.
        min_diff_holder : list of float
            Output parameter to store the minimum cost difference found.

        Returns
        -------
        int
            Index of the segment with minimum cost difference.
        """
        min_val = VERY_LARGE_COST
        loc = -1

        for i in range(regime0.num_segments):
            st = regime0.segments[i]["start"]
            length = regime0.segments[i]["duration"]
            costC0 = self.ap_seg.vit._viterbi(regime0.model, regime0.delta, st, length)
            costC1 = self.ap_seg.vit._viterbi(regime1.model, regime1.delta, st, length)
            diff = costC1 - costC0

            if min_val > diff:
                loc = i
                min_val = diff

        min_diff_holder[0] = min_val
        return loc

    def _remove_noise_aux(self, regime0, regime1, sample_ratio):
        """
        Auxiliary noise removal function during regime refinement.

        Parameters
        ----------
        regime0 : _Regime
            First candidate regime.
        regime1 : _Regime
            Second candidate regime.
        sample_ratio : float
            Threshold ratio for pruning based on global cost.
        """
        if sample_ratio == 0:
            return

        mprev = VERY_LARGE_COST
        th = self.ap_seg.costT * sample_ratio

        while mprev > regime0.num_segments + regime1.num_segments:
            mprev = regime0.num_segments + regime1.num_segments

            # Find minimum segment
            diff0 = [0.0]  # Use list to simulate pass-by-reference
            diff1 = [0.0]
            loc0 = self._find_min_diff(regime0, regime1, diff0)
            loc1 = self._find_min_diff(regime1, regime0, diff1)

            if diff0[0] < diff1[0]:
                min_val = diff0[0]
                id_min = 0
            else:
                min_val = diff1[0]
                id_min = 1

            # Check remove or not
            if min_val < th:
                if id_min == 0:
                    regime1.add_segment(
                        regime0.segments[loc0]["start"],
                        regime0.segments[loc0]["duration"],
                    )
                    regime0.remove_segment(loc0)
                else:
                    regime0.add_segment(
                        regime1.segments[loc1]["start"],
                        regime1.segments[loc1]["duration"],
                    )
                    regime1.remove_segment(loc1)


class _HMM:
    """
    Hidden Markov Model class.

    Attributes
    ----------
        n: Number of sequences
        k: Number of states
        pi: Initial state probabilities
        A: Transition matrix
        pi_denom: Denominator for pi (for incremental estimation)
        A_denom: Denominator for A (for incremental estimation)
        d: Dimension of observations
        mean: Means for Gaussian emissions
        var: Variances for Gaussian emissions
        sum_w: Total weighted count
        M2: M2 for incremental variance computation
    """

    def __init__(self, variance):

        self.default_variance = variance

        self.n = 0
        self.k = 0
        self.pi = None
        self.A = None
        self.pi_denom = 0.0
        self.A_denom = None

        self.d = 0
        self.mean = None
        self.var = None
        self.sum_w = None
        self.M2 = None

    def _set_up(self, k, d):
        """
        Initialize an HMM with given number of states and observation dimensions.

        Parameters
        ----------
        k : int
            Number of hidden states.
        d : int
            Dimensionality of observations (features per time step).
        """
        self.n = 0
        self.d = d
        self.k = k

        # Allocate arrays
        self.A = np.zeros((self.k, self.k), dtype=np.float64)
        self.pi = np.zeros(self.k, dtype=np.float64)
        self.mean = np.zeros((self.k, self.d), dtype=np.float64)
        self.var = np.zeros((self.k, self.d), dtype=np.float64)
        self.pi_denom = 0
        self.A_denom = np.zeros(self.k, dtype=np.float64)
        self.sum_w = np.zeros((self.k, self.d), dtype=np.float64)
        self.M2 = np.zeros((self.k, self.d), dtype=np.float64)

        # Random A matrix with EPSILON added, normalized row-wise
        self.A = np.random.rand(k, k) + EPSILON
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Random pi with EPSILON added, normalized
        self.pi = np.random.rand(k) + EPSILON
        self.pi /= self.pi.sum()

        # Random means, fixed variance
        self.mean = np.random.rand(k, d) * MAX_NORMALISATION_VALUE
        self.var.fill(self.default_variance)

    def reset(self, K, D):
        """
        Reset the HMM parameters to default values.

        Sets uniform initial state distribution and random emission parameters.

        Parameters
        ----------
        k : int
            Number of hidden states.
        d : int
            Dimensionality of observations (features per time step).
        """
        self.n = 0
        self.d = D
        self.k = K

        # Transition matrix A: random rows normalized
        self.A[:K, :K] = np.random.rand(K, K)
        self.A[:K, :K] /= self.A[:K, :K].sum(axis=1, keepdims=True)

        # Initial state probabilities pi: normalized
        self.pi[:K] = np.random.rand(K)
        self.pi[:K] /= self.pi[:K].sum()

        # Gaussian parameters: random means, fixed variance
        self.mean[:K, :D] = np.random.rand(K, D) * MAX_NORMALISATION_VALUE
        self.var[:K, :D] = self.default_variance

        # Reset accumulators
        self.pi_denom = 0.0
        self.A_denom[:K] = 0.0
        self.sum_w[:K, :D] = 0.0
        self.M2[:K, :D] = 0.0

    def copy_to(self, target):
        """
        Copy all parameters of this HMM into another HMM object.

        Parameters
        ----------
        target : _HMM
            Target HMM object to copy into.
        """
        target.d = self.d
        target.k = self.k
        target.n = self.n

        # Vectorized matrix/array copies
        target.A[: self.k, : self.k] = self.A[: self.k, : self.k]
        target.mean[: self.k, : self.d] = self.mean[: self.k, : self.d]
        target.var[: self.k, : self.d] = self.var[: self.k, : self.d]
        target.sum_w[: self.k, : self.d] = self.sum_w[: self.k, : self.d]
        target.M2[: self.k, : self.d] = self.M2[: self.k, : self.d]
        target.pi[: self.k] = self.pi[: self.k]
        target.A_denom[: self.k] = self.A_denom[: self.k]

        # Scalar copy
        target.pi_denom = self.pi_denom


class _BaumWelch:
    """
    Baum-Welch algorithm data class.

    Attributes
    ----------
        alpha: Forward probabilities [mx][k]
        beta: Backward probabilities [mx][k]
        gamma: State probabilities [n][mx][k]
        gamma_space: Space for gamma calculations
        xi: Transition probabilities [n][mx][k][k]
        scale: Scaling factors [mx]
        idx: Cluster indices for k-means [n][mx]
        chmm: Current HMM model
    """

    def __init__(self, ap_seg):
        self.ap_seg = ap_seg

        self.alpha = None
        self.beta = None
        self.gamma = None
        self.gamma_space = None
        self.xi = None
        self.scale = None
        self.idx = None
        self.chmm = _HMM(self.ap_seg.default_variance)

    def baum_welch(self, hmm, n, sequences, use_k_means):
        """
        Train a Hidden Markov Model (HMM) using the Baum-Welch algorithm.

        Parameters
        ----------
        hmm : _HMM
            Hidden Markov Model to train.
        n : int
            Number of input sequences.
        sequences : list of _Input
            List of input sequences.
        use_k_means : bool
            Whether to initialize parameters using K-means clustering.

        Returns
        -------
        float
            Final log-likelihood of the trained model.
        """
        length = 0
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        xi = self.xi
        scale = self.scale
        idx = self.idx

        if n == 0:
            raise ValueError("Estimation error (n == 0)")

        # If k==1, nothing to do (just run K-means)
        if hmm.k == 1:
            Kmeans(
                hmm,
                n,
                sequences,
                hmm.d,
                hmm.k,
                idx,
                self.ap_seg.max_k_means_iter,
                self.ap_seg,
            )
            return -1

        if use_k_means:
            # If initial stage and want k-means
            if hmm.n == 0:
                hmm.reset(hmm.k, hmm.d)
                Kmeans(
                    hmm,
                    n,
                    sequences,
                    hmm.d,
                    hmm.k,
                    idx,
                    self.ap_seg.max_k_means_iter,
                    self.ap_seg,
                )

        # Take absolute value of n
        n = int(abs(float(n)))

        prev_n = hmm.n
        hmm.n = n
        hmm.copy_to(self.chmm)

        # For each sequence
        Lsum = 0.0
        for r in range(hmm.n):
            obsevations = sequences[r].observations
            m = sequences[r].length

            # Cache emission probs for this sequence
            emission_probs = batch_pdf(hmm, hmm.k, m, obsevations)  # shape: (K, m)

            Lf = forward(hmm, obsevations, m, alpha, scale, emission_probs)
            backward(hmm, m, beta, scale, emission_probs)
            _compute_gamma(hmm, alpha, beta, gamma[r], m)
            _compute_xi(hmm, alpha, beta, xi[r], m, emission_probs)
            Lsum += Lf

        # Log likelihood
        Lpreb = Lsum

        # Baum-Welch iterations
        while True:
            # M-STEP: Update model parameters
            self.chmm.copy_to(hmm)
            self._compute_params(hmm, gamma, xi, sequences)

            # E-STEP: Re-estimate expectations
            Lsum = 0.0
            for r in range(hmm.n):
                obsevations = sequences[r].observations
                m = sequences[r].length

                # Cache emission probs for this sequence
                emission_probs = batch_pdf(hmm, hmm.k, m, obsevations)  # shape: (K, m)

                Lf = forward(hmm, obsevations, m, alpha, scale, emission_probs)
                backward(hmm, m, beta, scale, emission_probs)
                _compute_gamma(hmm, alpha, beta, gamma[r], m)
                _compute_xi(hmm, alpha, beta, xi[r], m, emission_probs)
                Lsum += Lf

            delta = Lpreb - Lsum
            Lpreb = Lsum
            length += 1

            # Check convergence
            if abs(delta) <= self.ap_seg.delta or length >= self.ap_seg.max_iter:
                break

        # For incremental fitting
        self._compute_params(hmm, gamma, xi, sequences)

        # Avoid numerical errors
        if math.isnan(Lsum):
            if self.ap_seg.verbose:
                sys.stdout.write("baumWelch: isnan, resetHMM...")  # noqa: T201
            hmm.reset(hmm.k, hmm.d)

        hmm.n = hmm.n + prev_n

        return Lsum

    def _compute_params(self, hmm, gamma, xi, sequences):
        """
        Re-estimate HMM parameters using collected gamma and xi statistics.

        Parameters
        ----------
        hmm : _HMM
            Hidden Markov Model to update.
        gamma : list of np.ndarray
            Posterior state probabilities (shape: [n][T][k]).
        xi : list of np.ndarray
            Posterior transition probabilities (shape: [n][T][k][k]).
        sequences : list of _Input
            List of input sequences.
        """
        k = hmm.k
        # Initial probability vector pi
        # (a) recover previous pi*N
        hmm.pi[:k] *= hmm.pi_denom  # restore weighted prior

        # (b) add new gamma
        gamma0_sum = sum(g[0] for g in gamma[:k])  # g[0] is shape (K,)
        hmm.pi[:k] += EPSILON + gamma0_sum[:k]

        # # (c) normalize, clip negatives and normalize
        np.clip(hmm.pi[:k], 0, None, out=hmm.pi[:k])
        hmm.pi_denom = np.sum(hmm.pi[:k])

        hmm.pi[:k] /= hmm.pi_denom

        # Transition matrix A
        # (a) recover previous A*N
        hmm.A[:k, :k] *= hmm.A_denom[:k, None]  # (K, 1)

        # (b) add new xi
        # (b) accumulate all xi transitions
        A_update = np.zeros((k, k), dtype=np.float64)
        for r in range(hmm.n):
            m_r = sequences[r].length
            A_update += np.sum(xi[r][: m_r - 1], axis=0) + EPSILON  # sum over t

        hmm.A[:k, :k] += A_update

        np.clip(hmm.A, 0.0, None, out=hmm.A)
        hmm.A_denom = np.sum(hmm.A, axis=1)
        hmm.A /= hmm.A_denom[:, None]

        # Weighted incremental computation for means and variances
        for i in range(k):
            for j in range(hmm.d):
                mean = hmm.mean[i][j]
                M2 = hmm.M2[i][j]
                sum_w = hmm.sum_w[i][j]

                if sum_w == 0.0:
                    mean = 0.0
                    M2 = 0.0

                for r in range(hmm.n):
                    obs = sequences[r].observations[:, j]  # shape: (T,)
                    w = gamma[r][:, i]  # shape: (T,)

                    for x, w_i in zip(obs, w):
                        tmp = w_i + sum_w + EPSILON
                        delta = x - mean
                        R = (delta * w_i) / tmp
                        mean += R
                        M2 += sum_w * delta * R
                        sum_w = tmp

                # Finalize variance
                var = M2 / sum_w if sum_w > 0 else 0.0
                var = min(self.ap_seg.max_variance, max(var, self.ap_seg.min_variance))

                hmm.mean[i][j] = mean
                hmm.M2[i][j] = M2
                hmm.sum_w[i][j] = sum_w
                hmm.var[i][j] = var

    def _set_up(self, n, m, k, d):
        """
        Initialise arrays for the Baum-Welch algorithm.

        Parameters
        ----------
        n : int
            Number of sequences.
        m : int
            Maximum sequence length.
        k : int
            Number of hidden states.
        d : int
            Observation dimensionality.
        """
        self.alpha = np.zeros((m, k), dtype=np.float64)
        self.beta = np.zeros((m, k), dtype=np.float64)
        self.scale = np.zeros(m, dtype=np.float64)
        self.idx = np.zeros((n, m), dtype=np.int32)

        # Gamma
        self.gamma = [None] * n
        for r in range(n):
            self.gamma[r] = np.zeros((m, k), dtype=np.float64)

        # Xi
        self.xi = [None] * n
        for r in range(n):
            self.xi[r] = [None] * m
            for t in range(m):
                self.xi[r][t] = np.zeros((k, k), dtype=np.float64)

        # For incremental EM
        self.chmm._set_up(k, d)


class _Viterbi:
    """
    Viterbi algorithm data class.

    Attributes
    ----------
        delta: Viterbi probabilities [m][k]
        psi: Backtracking indices [m][k]
        q: State sequence [m]
        piL: Log initial probabilities [k]
        AL: Log transition probabilities [k][k]
        biot: Log emission probabilities [k][m]
    """

    def __init__(self, ap_seg):
        """
        Initialize Viterbi decoder.

        Parameters
        ----------
        ap_seg : AutoPlaitSegmenter
            Reference to the parent AutoPlait segmenter for accessing data.
        """
        self.ap_seg = ap_seg

        self.delta = None
        self.psi = None
        self.q = None
        self.piL = None
        self.AL = None
        self.biot = None

    def _set_up(self, m, k):
        """
        Initialise arrays for Viterbi algorithm buffers.

        Parameters
        ----------
        m : int
            Maximum sequence length.
        k : int
            Number of hidden states.
        """
        self.delta = np.zeros((m, k), dtype=np.float64)
        self.psi = np.zeros((m, k), dtype=np.int32)
        self.q = np.zeros(m, dtype=np.int32)
        self.piL = np.zeros(k, dtype=np.float64)
        self.AL = np.zeros((k, k), dtype=np.float64)
        self.biot = np.zeros((k, m), dtype=np.float64)

    def _viterbi(self, hmm, delta, start_idx, length):
        """
        Run the Viterbi algorithm on a subsequence, including switching penalties.

        Parameters
        ----------
        hmm : _HMM
            Hidden Markov Model used for decoding.
        delta : float
            Switch penalty probability (must be 0.0 < delta < 1.0).
        start_idx : int
            Start index of the subsequence.
        length : int
            Length of the subsequence.

        Returns
        -------
        float
            Coding cost (in bits) for the best path.

        Raises
        ------
        ValueError
            If `delta` is not between 0 and 1.
        """
        Lh = self.log_space_viterbi(
            hmm, length, self.ap_seg.x.observations[start_idx : start_idx + length]
        )

        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be 0.0 < delta < 1.0")

        Lh += math.log(delta)  # Switch
        Lh += (length - 1) * math.log(1.0 - delta)  # Else (stay)
        costC = -Lh / math.log(2.0)

        return costC

    def log_space_viterbi(self, hmm, length, observations):
        """
        Decode the most likely state sequence using the Viterbi algorithm in log-space.

        Parameters
        ----------
        hmm : _HMM
            Hidden Markov Model.
        length : int
            Length of the sequence.
        observations : np.ndarray
            Observation sequence (shape: [m, d]).

        Returns
        -------
        float
            Log-likelihood of the most probable path.

        Raises
        ------
        ValueError
            If no valid path can be found (numerical issues).
        """
        k = hmm.k

        if length == 0:
            return 0

        self.precompute_log_hmm(hmm, length, observations)

        delta = self.delta
        psi = self.psi
        q = self.q
        piL = self.piL
        AL = self.AL
        biot = self.biot

        # compute delta (t==0)
        delta[0, :k] = piL[:k] + biot[:k, 0]
        psi[0, :k] = 0

        # compute delta (t>0)
        for t in range(1, length):
            deltax = -VERY_LARGE_COST
            for j in range(hmm.k):
                maxval = -VERY_LARGE_COST
                maxvalind = 0
                for i in range(hmm.k):
                    val = delta[t - 1][i] + AL[i][j]
                    if val > maxval:
                        maxval = val
                        maxvalind = i
                delta[t][j] = maxval + biot[j][t]
                psi[t][j] = maxvalind
                if deltax < delta[t][j]:
                    deltax = delta[t][j]

        # final likelihood
        Lh = -VERY_LARGE_COST
        q[length - 1] = -1
        for i in range(hmm.k):
            if delta[length - 1][i] > Lh:
                Lh = delta[length - 1][i]
                q[length - 1] = i

        # avoid error
        if q[length - 1] == -1:
            raise ValueError("Cannot compute log Viterbi path")

        # check viterbi path
        for t in range(length - 2, -1, -1):
            q[t] = psi[t + 1][q[t + 1]]

        return Lh

    def precompute_log_hmm(self, hmm, length, observations):
        """
        Precompute log-probabilities for HMM emissions and transitions.

        Parameters
        ----------
        hmm : _HMM
            Hidden Markov Model.
        length : int
            Maximum length of the observation sequence.
        observations : np.ndarray
            Observation sequence (shape: [m, d]).
        """
        self.piL[: hmm.k] = np.log(hmm.pi[: hmm.k] + EPSILON)

        for i in range(hmm.k):
            for j in range(hmm.k):
                self.AL[i][j] = math.log(hmm.A[i][j] + EPSILON)

        length = min(length, observations.shape[0])
        self.biot[: hmm.k, :length] = batch_log_pdf(
            hmm, hmm.k, length, observations[:length]
        )


# ------------------------------
#      OBSERVATION FUNCTIONS
# ------------------------------
def _split_sequence(input_seq, start_idx, length, subseq):
    """
    Split a subsequence from a parent input sequence.

    Parameters
    ----------
    input_seq : _Input
        Parent input sequence.
    start_idx : int
        Start index for the split.
    length : int
        Length of the split.
    subseq : _Input
        Target input structure to store the split subsequence.
    """
    if start_idx < 0:
        start_idx = 0
    if input_seq.length < start_idx + length:
        length = input_seq.length - start_idx

    # In Python, we need to create a view of the array rather than pointer arithmetic
    subseq.observations = input_seq.observations[start_idx : start_idx + length]
    subseq.length = length
    subseq.parent = input_seq.id
    subseq.start_idx = start_idx
    subseq.tag = (
        f"[{subseq.id}] {input_seq.tag} [{start_idx}-{start_idx + length}]({length})"
    )


# ------------------------------
#      BAUM-WELCH FUNCTIONS
# ------------------------------
def forward(hmm, observations, m, alpha, scale, emission_probs):
    """
    Run the Forward algorithm for a single sequence.

    Parameters
    ----------
    hmm : _HMM
        Hidden Markov Model.
    observations : np.ndarray
        Observation sequence (shape: [T, d]).
    m : int
        Length of sequence.
    alpha : np.ndarray
        Forward probability matrix (shape: [T, k]).
    scale : np.ndarray
        Scaling factors to prevent underflow.
    emission_probs : np.ndarray
        Precomputed emission probabilities (shape: [k, T]).

    Returns
    -------
    float
        Log-likelihood of the observation sequence.
    """
    # Initialize scale
    scale[:m] = 0.0

    O0 = np.asarray([observations[0]])  # shape: (1, D)
    obs_probs = batch_pdf(hmm, hmm.k, m, O0)[:, 0]  # shape: (K,)
    alpha[0, : hmm.k] = hmm.pi[: hmm.k] * obs_probs
    scale[0] = alpha[0].sum()

    alpha[0, : hmm.k] /= scale[0]

    # Induction
    for t in range(m - 1):
        # Matrix multiply: alpha[t] (1, K) x A (K, K) -> (K,)
        alpha_t1 = alpha[t, : hmm.k] @ hmm.A[: hmm.k, : hmm.k]  # shape: (K,)
        # Multiply by emission probabilities at time t+1
        alpha[t + 1, : hmm.k] = alpha_t1 * emission_probs[: hmm.k, t + 1]

        # Normalize and store scale
        scale[t + 1] = alpha[t + 1, : hmm.k].sum()
        alpha[t + 1, : hmm.k] /= scale[t + 1]

    # Termination
    return np.sum(np.log(scale[:m]))


def backward(hmm, m, beta, scale, emission_probs):
    """
    Run the Backward algorithm for a single sequence.

    Parameters
    ----------
    hmm : _HMM
        Hidden Markov Model.
    m : int
        Length of sequence.
    beta : np.ndarray
        Backward probability matrix (shape: [T, k]).
    scale : np.ndarray
        Scaling factors from the Forward pass.
    emission_probs : np.ndarray
        Precomputed emission probabilities (shape: [k, T]).

    Returns
    -------
    float
        Log-likelihood of the observation sequence (should match forward pass).
    """
    # Init (t=m-1)
    beta[m - 1, : hmm.k] = 1.0 / scale[m - 1]

    # Induction
    for t in range(m - 2, -1, -1):
        # Emission at time t+1 across all states (shape: K,)
        emis = emission_probs[: hmm.k, t + 1]

        # Elementwise multiply: emission * beta[t+1]
        weighted = emis * beta[t + 1, : hmm.k]  # shape: (K,)

        # Matrix multiply: A (KxK) x weighted (K,) → (K,)
        beta[t, : hmm.k] = hmm.A[: hmm.k, : hmm.k] @ weighted

        # Normalize by scale[t]
        beta[t] /= scale[t]

    # Log likelihood (same as Forward)
    return np.sum(np.log(scale[:m]))


def _compute_gamma(phmm, alpha, beta, gamma, m):
    """
    Compute posterior state probabilities (gamma) for a sequence.

    Parameters
    ----------
    phmm : _HMM
        Hidden Markov Model.
    alpha : np.ndarray
        Forward probabilities.
    beta : np.ndarray
        Backward probabilities.
    gamma : np.ndarray
        Output gamma array (shape: [T, k]).
    m : int
        Length of sequence.
    """
    product = alpha[:m, : phmm.k] * beta[:m, : phmm.k]  # (m, K)
    gamma[:m, : phmm.k] = product / product.sum(axis=1, keepdims=True)


def _compute_xi(phmm, alpha, beta, xi, m, emission_probs):
    """
    Compute posterior transition probabilities (xi) for a sequence.

    Parameters
    ----------
    phmm : _HMM
        Hidden Markov Model.
    alpha : np.ndarray
        Forward probabilities.
    beta : np.ndarray
        Backward probabilities.
    xi : np.ndarray
        Output xi array (shape: [T, k, k]).
    m : int
        Length of sequence.
    emission_probs : np.ndarray
        Precomputed emission probabilities (shape: [k, T]).
    """
    k = phmm.k
    A = phmm.A[:k, :k]  # shape: (K, K)

    for t in range(m - 1):
        # Shapes:
        # alpha[t]: (K,)
        # A: (K, K)
        # emission_probs[:, t+1]: (K,)
        # beta[t+1]: (K,)

        # Broadcast to shape (K, K)
        outer = (
            alpha[t, :k][:, None]
            * A
            * emission_probs[:k, t + 1][None, :]
            * beta[t + 1, :k][None, :]
        )

        # Normalize
        xi[t] = outer / np.sum(outer)


# ------------------------------
#         SAMPLING FUNCTIONS
# ------------------------------
def fixed_sampling(current_regime, regime0, regime1, length):
    """
    Perform fixed (non-random) sampling of two segments.

    Parameters
    ----------
    current_regime : _Regime
        Source regime to sample from.
    regime0 : _Regime
        First output regime.
    regime1 : _Regime
        Second output regime.
    length : int
        Length of each sampled segment.
    """
    # Initialize segments
    regime0.reset()
    regime1.reset()

    # Segment regime0
    loc = 0 % current_regime.num_segments
    r = current_regime.segments[loc]["start"]
    regime0.add_segment(r, length)

    # Segment regime1
    loc = 1 % current_regime.num_segments
    r = current_regime.segments[loc]["start"] + int(
        current_regime.segments[loc]["duration"] / 2
    )
    regime1.add_segment(r, length)


def uniform_sampling(regime0, regime1, length, n1, n2, U):
    """
    Perform uniform sampling of two initial segments.

    Parameters
    ----------
    regime0 : _Regime
        First candidate regime.
    regime1 : _Regime
        Second candidate regime.
    length : int
        Length of sampled segments.
    n1 : int
        Index of first sample.
    n2 : int
        Index of second sample.
    U : _Regime
        Source regime to sample from.
    """
    # Initialize segments
    regime0.reset()
    regime1.reset()

    i = int(n1 % U.num_segments)
    j = int(n2 % U.num_segments)

    st0 = U.segments[i]["start"]
    st1 = U.segments[j]["start"]

    # If overlapped, then ignore
    if abs(st0 - st1) < length:
        return

    regime0.add_segment(st0, length)
    regime1.add_segment(st1, length)


def uniform_set(current_regime, length, trial, U):
    """
    Create a set of uniformly spaced subsequences for sampling.

    Parameters
    ----------
    current_regime : _Regime
        Source regime to sample from.
    length : int
        Length of each sample segment.
    trial : int
        Number of samples to generate.
    U : _Regime
        Regime to populate with sampled segments.
    """
    slideW = int(math.ceil((current_regime.total_length - length) / trial))

    # Create uniform blocks
    U.reset()

    for i in range(current_regime.num_segments):
        if U.num_segments >= trial:
            return

        st = current_regime.segments[i]["start"]
        ed = st + current_regime.segments[i]["duration"]

        for j in range(trial):
            next_pos = st + j * slideW

            if next_pos + length > ed:
                st = ed - length
                if st < 0:
                    st = 0
                U.add_segment_with_overlap(st, length)
                break

            U.add_segment_with_overlap(next_pos, length)


# ------------------------------
#         KMEANS FUNCTIONS
# ------------------------------
def compute_cluster_stats(data, labels, k, d, ap_seg):
    """
    Compute per-cluster mean and variance (vectorized).

    Parameters
    ----------
        data: np.ndarray (N, D)
        labels: np.ndarray (N,)
        k: number of clusters
        d: dimensionality
    Returns:
        means: (k, D)
        vars: (k, D)
    """
    means = np.zeros((k, d))
    vars_ = np.zeros((k, d))

    for j in range(k):
        mask = labels == j
        cluster_points = data[mask]

        if len(cluster_points) == 0:
            # Fallback to random mean/var if empty
            means[j] = np.random.rand(d)
            vars_[j] = ap_seg.min_variance
            continue

        mean = np.mean(cluster_points, axis=0)
        var = np.var(cluster_points, axis=0, ddof=1)  # sample variance

        means[j] = mean
        vars_[j] = np.clip(var + EPSILON, ap_seg.min_variance, ap_seg.max_variance)

    return means, vars_


def Kmeans(phmm, n, xlst, d, k, idx, max_iter, ap_seg):
    """Perform K-means clustering.

    Parameters
    ----------
    phmm -- HMM model object with mean and var attributes
    n -- number of sequences
    xlst -- input data (list of Input objects with O and m attributes)
    d -- dimension of data
    k -- number of clusters
    idx -- cluster assignments (2D array[n][m])
    """
    # if k==1, nothing to do
    if k == 1:
        # Flatten all observations
        X = np.vstack([x.observations for x in xlst[:n]])

        # Assign all to cluster 0
        for r in range(n):
            idx[r][: xlst[r].length] = 0

        # Compute global mean and variance
        phmm.mean[0] = np.mean(X, axis=0)
        var = np.var(X, axis=0, ddof=1)
        phmm.var[0] = np.clip(var + EPSILON, ap_seg.min_variance, ap_seg.max_variance)

        return

    # Flatten data
    X = np.vstack([x.observations for x in xlst[:n]])  # shape: (N_total, D)

    kmeans = KMeans(
        n_clusters=k, init="random", n_init=1, max_iter=max_iter, random_state=42
    )

    labels = kmeans.fit_predict(X)

    offset = 0
    for r in range(n):
        for i in range(xlst[r].length):
            idx[r][i] = labels[offset]
            offset += 1

    # Final calculation of means and variances
    phmm.mean[:k], phmm.var[:k] = compute_cluster_stats(X, labels, k, d, ap_seg)


# ------------------------------
#         MATH FUNCTIONS
# ------------------------------


def log_2(x):
    """Log base 2 of x."""
    return math.log(x) / math.log(2.0)


def log_s(x):
    """Compute 2*log_2(x) + 1."""
    return 2.0 * log_2(x) + 1.0


def batch_log_pdf(hmm, k, m, observations):
    observations = observations[:m]  # (T, D)
    mean = hmm.mean[:k]  # (K, D)
    var = np.abs(hmm.var[:k])  # (K, D)

    # Expand dimensions for broadcasting: (K, T, D)
    observations_exp = observations[None, :, :]  # (1, T, D)
    mean_exp = mean[:, None, :]  # (K, 1, D)
    var_exp = var[:, None, :]  # (K, 1, D)

    # Gaussian PDF calculation
    diff_sq = (observations_exp - mean_exp) ** 2
    denom = np.sqrt(2 * np.pi * var_exp)
    exponent = -diff_sq / (2 * var_exp)

    p = np.exp(exponent) / denom

    # Clamp to [EPSILON, ONE_MINUS_EPSILON]
    p = np.clip(p, EPSILON, ONE_MINUS_EPSILON)

    # log(EPSILON + p)
    log_probs = np.log(EPSILON + p)

    # Sum over dimensions (D), result is (K, T)
    log_sums = log_probs.sum(axis=2)

    # Final clamp: if sum < log(EPSILON), set to log(EPSILON)
    return np.maximum(log_sums, LOG_EPSILON)


def batch_pdf(hmm, k, m, observations):
    return np.exp(batch_log_pdf(hmm, k, m, observations))
