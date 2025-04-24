"""
autoplait.py - Python version of autoplait.c
Originally written by Yasuko Matsubara
Translated to Python in 2025

This is the main entry point for the AutoPlait algorithm.
"""

import math
import random
import sys
import warnings

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

# ------------------------------
#     autoplait_header_py.py
# ------------------------------
"""
autoplait_header.py - Python version of autoplait.h
Originally written by Yasuko Matsubara
Translated to Python in 2025
"""

class AutoPlaitSegmenter(BaseSegmenter):
    """
    AutoPlait workspace structure.

    Attributes:
        x: Input sequence
        maxc: Maximum number of clusters
        maxseg: Maximum number of segments
        d: Dimension
        lmax: Maximum length
        cps: CutPointSearch structure
        baum: BAUM structure for EM algorithm
        q: Viterbi path
        vit, vit2: VITERBI structures
        x_tmp: Temporary input
        s: Current segment
        C: Candidate stack
        Opt: Optimal segments stack
        S: Segments stack
        costT: Total cost
        U: Uniform sampling
    """

    _tags = {
        "returns_dense": True,
        "fit_is_empty": False,
        "capability:multivariate": True
    }

    def __init__(self,
                 max_segments=100,
                 max_sequence_length=10000000,
                 min_k=1,
                 max_k=16,
                 num_samples=10,
                 max_baumn_segments=3,
                 delta=1,
                 max_iter=10,
                 max_k_means_iter=100,
                 min_infer_iter=3,
                 max_infer_iter=10,
                 variance=10,
                 max_variance=float('inf'),
                 min_variance=None,
                 segment_sample_ratio=0.03,
                 regime_sample_ratio=0.03,
                 samplimg_lm=0.1,
                 seed=None,
                 normalise=True,
                 verbose=False):

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
        self.variance = variance
        self.max_variance = max_variance
        self.min_variance = min_variance
        self.segment_sample_ratio = segment_sample_ratio
        self.regime_sample_ratio = regime_sample_ratio
        self.samplimg_lm = samplimg_lm

        self.x = None
        self.maxc = 0
        self.maxseg = 0
        self.d = 0
        self.lmax = max_sequence_length
        self.cps = CPS(verbose)
        self.baum = BAUM(self.variance, self.delta, self.max_iter, self.max_k_means_iter)
        self.q = None
        self.vit = VITERBI()
        self.vit2 = VITERBI()
        self.x_tmp = None
        self.s = None
        self.C = []
        self.Opt = []
        self.S = []
        self.costT = 0.0
        self.U = Regime(self.max_segments, self.min_k, self.max_k, self.variance)

        self.verbose = verbose
        self.axis = 0
        self.seed = seed
        self._do_normalise = normalise
        random.seed(seed)

        super().__init__(axis=0)

    def _set_up(self):
        """
            Allocate memory for AutoPlait structures.

            Parameters:
            pws -- AutoPlait workspace
            """
        global ws
        ws = self

        if self.verbose: sys.stderr.write("memory allocation...\n")

        ws.maxc = int(math.log(ws.lmax) + 20)
        ws.maxseg = self.max_segments  # int(ws.lmax * 0.1)

        # Allocate Viterbi
        AllocViterbi(ws.vit, ws.maxseg, ws.lmax, self.max_k)
        AllocViterbi(ws.vit2, ws.maxseg, ws.lmax, self.max_k)
        ws.q = np.zeros(ws.lmax, dtype=np.int32)#ivector(0, ws.lmax)
        ws.cps._set_up(self.max_k, ws.lmax)

        # Allocate Baum
        n = ws.maxseg
        if n > self.max_baumn_segments:
            n = self.max_baumn_segments

        ws.x_tmp = [Input() for _ in range(n)]
        AllocBaum(ws.baum, n, ws.lmax, self.max_k, ws.d)

        # Allocate segbox
        ws.s = [Regime(self.max_segments, self.min_k, self.max_k, self.variance) for _ in range(ws.maxc)]
        for i in range(ws.maxc):
            _allocSegBox(ws.s[i])

        for i in range(ws.maxc):
            ws.S.append(ws.s[i])

        # For uniform sampling
        _allocSegBox(ws.U)

    def _fit(self, X, y=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.d = X.shape[1]
        inp = Input()
        inp.m = min(X.shape[0], self.lmax)
        inp.id = 0
        inp.O = X
        self.x = inp
        self.lmax = self.x.m
        self._set_up()
        if self._do_normalise: self.normalise()

    def _predict(self, X):
        """
        Main entry point for AutoPlait algorithm.

        Parameters:
        pws -- AutoPlait workspace
        """
        global ws
        ws = self

        if self.verbose: sys.stdout.write("---------\n")
        if self.verbose: sys.stdout.write("r|m|Cost \n")
        if self.verbose: sys.stdout.write("---------\n")

        self._plait()
        output = []
        for i in range(len(self.Opt)):
            regime = self.Opt[i]
            for j in range(regime.num_segments):
                output.append(((regime.segments[j]['start'] + regime.segments[j]['duration'] - 1), i))
        output = sorted(output, key=lambda x: x[0])
        cps = [x[0] for x in output][:-1]
        self.regimes = [x[1] for x in output]
        return np.array(cps)

    def _plait(self):
        """Main AutoPlait algorithm implementation."""
        # Initialize current_regime (X[0:m])
        current_regime = _getS("", "")
        current_regime.costT = VERY_LARGE_COST
        current_regime.add_segment(0, ws.x.m)
        current_regime.estimate_parameters()
        ws.C.append(current_regime)

        while True:
            ws.costT = _MDLtotal(ws.Opt, ws.C, self.verbose)
            if not ws.C:
                break
            current_regime = ws.C.pop()

            # Create new segment sets
            regime0 = _getS(current_regime.label, "0")
            regime1 = _getS(current_regime.label, "1")

            # Try to split regime: current_regime->(regime0,regime1)
            _regimeSplit(current_regime, regime0, regime1)
            costT_regime01 = regime0.costT + regime1.costT

            # Split or not
            if costT_regime01 + current_regime.costT * self.regime_sample_ratio < current_regime.costT:
                ws.C.append(regime0)
                ws.C.append(regime1)
                ws.S.append(current_regime)
            else:
                ws.Opt.append(current_regime)
                ws.S.append(regime0)
                ws.S.append(regime1)

    def normalise(self):
        """Z-normalize sequences (mean=0, std=1)."""
        if self.verbose: sys.stderr.write("Z-normalization... \n")

        for d in range(self.d):
            mean = 0.0
            std = 0.0
            cnt = 0

            x = self.x
            cnt += x.m
            for j in range(x.m):
                mean += x.O[j][d]

            mean /= cnt

            # Compute std
            x = self.x
            for j in range(x.m):
                std += (x.O[j][d] - mean) ** 2

            std = math.sqrt(std / cnt)

            # Normalize
            x = self.x
            for j in range(x.m):
                x.O[j][d] = MAX_NORMALISATION_VALUE * (x.O[j][d] - mean) / std

class Input:
    """
    Input data class for HMM.

    Attributes:
        id: Sequence identifier
        tag: String tag for the sequence
        parent: Parent sequence id
        st: Start position in parent sequence
        O: Observation sequence [m][d]
        m: Length of sequence
        pid: Pattern id
    """

    def __init__(self):
        self.id = 0
        self.tag = ""
        self.parent = 0
        self.st = 0
        self.O = None
        self.m = 0
        self.pid = 0

class Regime:
    """
    A structure representing a segmented regime of a time series.

    Attributes:
        segments (np.ndarray): Structured NumPy array of segments, each with 'start' and 'duration'.
        nSb (int): Number of active segments.
        total_length (int): Total combined duration of all segments.
        costT (float): Total cost including model and encoding.
        costC (float): Cost of encoding the data.
        optimal (bool): Whether this regime is considered optimal.
        label (str): Label for the regime.
        model (HMM): Hidden Markov Model associated with this regime.
        delta (float): Proportion of time spent in this regime relative to total sequence length.
    """

    def __init__(self, max_segments, min_k, max_k, variance, label=''):

        self.max_segments = max_segments
        self.min_k = min_k
        self.max_k = max_k
        self.default_variance = variance


        segment_dtype = np.dtype([('start', np.int32), ('duration', np.int32)])
        self.segments = np.zeros(self.max_segments, dtype=segment_dtype)
        self.num_segments = 0
        self.total_length = 0
        self.costT = 0.0
        self.costC = 0.0
        self.optimal = False
        self.label = label
        self.model = HMM(self.default_variance)
        self.delta = 0.0

    def reset(self) -> None:
        """
        Resets the regime to an empty state, clearing all segments and resetting costs.
        """
        self.num_segments = 0
        self.total_length = 0
        self.costC = float('inf')
        self.costT = float('inf')

    def add_segment(self, start: int, duration: int) -> None:
        """
        Adds a segment to the regime and automatically removes any overlaps.

        Args:
            start (int): Start index of the new segment.
            duration (int): Duration of the new segment.

        Raises:
            RuntimeError: If the number of segments exceeds MAX_SEGMENTS.
        """
        if duration <= 0:
            return
        start = max(0, start)

        # Find insertion point
        loc = 0
        while loc < self.num_segments and self.segments[loc]['start'] <= start:
            loc += 1

        # Shift elements to make space
        if self.num_segments >= self.max_segments:
            raise RuntimeError("Too many segments")

        if loc < self.num_segments:
            self.segments[loc + 1:self.num_segments + 1] = self.segments[loc:self.num_segments]

        # Insert new segment
        self.segments[loc]['start'] = start
        self.segments[loc]['duration'] = duration
        self.num_segments += 1

        self._remove_overlap()
        self.total_length = self.segments[:self.num_segments]['duration'].sum()

    def _remove_overlap(self) -> None:
        """
        Merges overlapping or adjacent segments in-place.
        """
        i = 0
        while i < self.num_segments - 1:
            starts = self.segments[:self.num_segments]['start']
            durations = self.segments[:self.num_segments]['duration']
            ends = starts + durations

            seg0_start = starts[i]
            seg0_end = ends[i]
            seg1_start = starts[i + 1]
            seg1_end = ends[i + 1]

            if seg0_end + 1 >= seg1_start:
                # Merge seg0 and seg1
                self.segments[i]['duration'] = max(seg0_end, seg1_end) - seg0_start
                self.remove_segment(i + 1)
                # Don't advance i
            else:
                i += 1

    def remove_segment(self, index: int) -> None:
        """
        Removes a segment at the specified index from the regime.

        Args:
            index (int): Index of the segment to remove.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index >= self.num_segments:
            raise IndexError(f"remove_segment: index {index} out of bounds (nSb={self.num_segments})")

        self.total_length -= self.segments[index]['duration']

        if index < self.num_segments - 1:
            self.segments[index:self.num_segments - 1] = self.segments[index + 1:self.num_segments]

        self.segments[self.num_segments - 1] = (0, 0)  # Optional clear
        self.num_segments -= 1

    def add_segment_with_overlap(self, start: int, duration: int) -> None:
        """
        Adds a segment to the regime without removing overlaps.

        Args:
            start (int): Start index of the segment.
            duration (int): Duration of the segment.

        Raises:
            RuntimeError: If the number of segments exceeds MAX_SEGMENTS.
        """
        if duration <= 0:
            return
        if self.num_segments >= self.max_segments:
            raise RuntimeError("Exceeded maximum number of segments")

        self.segments[self.num_segments]['start'] = start
        self.segments[self.num_segments]['duration'] = duration
        self.num_segments += 1
        self.total_length += duration

    def copy_to(self, target: 'Regime') -> None:
        """
        Copies all attributes and segment data to another Regime object.

        Args:
            target (Regime): The target Regime instance to copy into.
        """
        # Copy active segment data using slicing
        target.segments[:self.num_segments] = self.segments[:self.num_segments]

        target.num_segments = self.num_segments
        target.total_length = self.total_length
        target.costT = self.costT
        target.costC = self.costC
        target.delta = self.delta
        target.label = self.label

    def max_segment_index(self) -> int:
        """
        Returns the index of the longest segment.

        Returns:
            int: Index of the segment with maximum duration, or -1 if no segments exist.
        """
        if self.num_segments == 0:
            return -1
        return int(np.argmax(self.segments[:self.num_segments]['duration']))

    def estimate_parameters(self) -> None:
        """
        Estimates an optimal HMM by testing values of k from MIN_K to MAX_K.
        Selects the model that minimizes the MDL.
        """
        self.costT = float('inf')
        optk = self.min_k

        for k in range(self.min_k, self.max_k + 1):
            previous_cost = self.costT
            self.estimate_hmm_k(k)
            self.compute_likelihood_mdl()

            if self.costT > previous_cost:
                optk = max(self.min_k, k - 1)
                break

        optk = min(self.max_k, max(self.min_k, optk))

        self.estimate_hmm_k(optk)
        self.compute_likelihood_mdl()

    def estimate_hmm_k(self, k: int) -> None:
        """
        Estimates the HMM parameters for a given number of states.

        Args:
            k (int): Number of HMM states to estimate.
        """
        k = min(self.max_k, max(self.min_k, k))  # Clamp k to [MIN_K, MAX_K]

        num_segments = min(self.num_segments, ws.max_baumn_segments)
        for i in range(num_segments):
            Split(ws.x, self.segments[i]['start'], self.segments[i]['duration'], ws.x_tmp[i])

        self.model.k = k
        self.model.n = 0
        self.model.reset(k, ws.d)

        BaumWelch(self.model, num_segments, ws.x_tmp, ws.baum, wantKMEANS=True)
        self.delta = self.num_segments / self.total_length if self.total_length else 0.0

    def mdl(self) -> float:
        """
        Calculates the Minimum Description Length (MDL) of the regime.

        Returns:
            float: The total MDL cost including encoding and model complexity.
        """
        k = self.model.k
        d = self.model.d
        m = self.total_length

        durations = self.segments[:self.num_segments]['duration']
        costLen = np.log2(durations).sum()
        costLen += m * np.log2(k)

        costM = FLOATING_POINT_CONSTANT * (k + k * k + 2 * k * d) + log_s(k)
        return self.costC + costLen + costM

    def compute_likelihood_mdl(self) -> None:
        """
        Computes the log-likelihood cost for all segments using the Viterbi algorithm,
        then computes the full MDL.
        """
        if self.num_segments == 0:
            self.costC = float('inf')
            self.costT = float('inf')
            return

        starts = self.segments[:self.num_segments]['start']
        durations = self.segments[:self.num_segments]['duration']

        costs = np.fromiter(
            (_viterbi(self.model, self.delta, int(st), int(dur), ws.vit)
             for st, dur in zip(starts, durations)),
            dtype=float,
            count=self.num_segments
        )
        self.costC = costs.sum()
        self.costT = self.mdl()

    def largest_segment(self) -> None:
        """
        Reduces the regime to its single largest segment.
        """
        index = self.max_segment_index()
        if index == -1:
            return

        self.reset()
        self.add_segment(
            self.segments[index]['start'],
            self.segments[index]['duration']
        )

class CPS:
    """
    CutPointSearch structure.

    Attributes:
        Pu, Pv, Pi, Pj: Probability arrays
        Su, Sv, Si, Sj: State arrays
        nSu, nSv, nSi, nSj: Counts
    """

    def __init__(self, verbose):
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

        self.verbose = verbose

    def _set_up(self, maxk, maxlen):
        """
        Allocate memory for Cut Point Search data structures.

        Parameters:
        cps -- CPS structure
        maxk -- Maximum number of states
        maxlen -- Maximum sequence length
        """
        if self.verbose: sys.stderr.write(f"alloc cut point search...(k:{maxk},len:{maxlen})\n")
        self.Pu = np.zeros(maxk, dtype=np.float64)
        self.Pv = np.zeros(maxk, dtype=np.float64)
        self.Pi = np.zeros(maxk, dtype=np.float64)
        self.Pj = np.zeros(maxk, dtype=np.float64)
        self.Su = np.zeros((maxk, maxlen), dtype=np.int32)
        self.Sv = np.zeros((maxk, maxlen), dtype=np.int32)
        self.Si = np.zeros((maxk, maxlen), dtype=np.int32)
        self.Sj = np.zeros((maxk, maxlen), dtype=np.int32)
        self.nSu = np.zeros(maxk, dtype=np.int32)
        self.nSv = np.zeros(maxk, dtype=np.int32)
        self.nSi = np.zeros(maxk, dtype=np.int32)
        self.nSj = np.zeros(maxk, dtype=np.int32)

def _allocSegBox(s):
    """
    Allocate memory for segment box.

    Parameters:
    s -- Segment box
    n -- Maximum number of sub-segments
    """

    s.num_segments = 0
    s.optimal = False
    InitHMM(s.model, ws.max_k, ws.d)

def _getS(parent, label):
    """
    Get a segment box.

    Parameters:
    parent -- Parent label
    label -- Segment label

    Returns:
    Segment box
    """

    s = ws.S.pop()
    if s is None:
        raise ValueError('Too small maxc')

    s.reset()
    s.model.reset(ws.max_k, ws.d)
    s.label = f"{parent}{label}"
    s.delta = 1.0 / float(ws.lmax)

    return s

def _findMinDiff(regime0, regime1, diffp):
    """
    Find minimum difference between segments.

    Parameters:
    regime0 -- First segment
    regime1 -- Second segment
    diffp -- Reference to store difference value

    Returns:
    Location of minimum difference
    """
    min_val = VERY_LARGE_COST
    loc = -1

    for i in range(regime0.num_segments):
        st = regime0.segments[i]['start']
        length = regime0.segments[i]['duration']
        costC0 = _viterbi(regime0.model, regime0.delta, st, length, ws.vit)
        costC1 = _viterbi(regime1.model, regime1.delta, st, length, ws.vit)
        diff = costC1 - costC0

        if min_val > diff:
            loc = i
            min_val = diff

    diffp[0] = min_val
    return loc


def _scanMinDiff(current_regime, regime0, regime1):
    """
    Scan for minimum difference.

    Parameters:
    current_regime -- Source segment
    regime0 -- First segment
    regime1 -- Second segment

    Returns:
    Minimum difference cost
    """
    diff = [0.0]  # Use list to simulate pass-by-reference
    loc0 = _findMinDiff(regime0, regime1, diff)
    loc1 = _findMinDiff(regime1, regime0, diff)

    if loc0 == -1 or loc1 == -1:
        return VERY_LARGE_COST

    tmp0 = _getS("", "")
    tmp1 = _getS("", "")
    tmp0.add_segment(regime0.segments[loc0]['start'], regime0.segments[loc0]['duration'])
    tmp1.add_segment(regime1.segments[loc1]['start'], regime1.segments[loc1]['duration'])
    tmp0.estimate_hmm_k(tmp0.min_k)
    tmp1.estimate_hmm_k(tmp1.min_k)
    costC = CPSearch(current_regime, tmp0, tmp1, ws)
    ws.S.append(tmp0)
    ws.S.append(tmp1)

    return costC


def _removeNoise_aux(regime0, regime1, per):
    """
    Auxiliary function for noise removal.

    Parameters:
    current_regime -- Source segment
    regime0 -- First segment
    regime1 -- Second segment
    per -- Percentage threshold
    """
    if per == 0:
        return

    mprev = VERY_LARGE_COST
    th = ws.costT * per

    while mprev > regime0.num_segments + regime1.num_segments:
        mprev = regime0.num_segments + regime1.num_segments

        # Find minimum segment
        diff0 = [0.0]  # Use list to simulate pass-by-reference
        diff1 = [0.0]
        loc0 = _findMinDiff(regime0, regime1, diff0)
        loc1 = _findMinDiff(regime1, regime0, diff1)

        if diff0[0] < diff1[0]:
            min_val = diff0[0]
            id_min = 0
        else:
            min_val = diff1[0]
            id_min = 1

        # Check remove or not
        if min_val < th:
            if id_min == 0:
                regime1.add_segment(regime0.segments[loc0]['start'], regime0.segments[loc0]['duration'])
                regime0.remove_segment(loc0)
            else:
                regime0.add_segment(regime1.segments[loc1]['start'], regime1.segments[loc1]['duration'])
                regime1.remove_segment(loc1)


def _removeNoise(current_regime, regime0, regime1):
    """
    Remove noise from segments.

    Parameters:
    current_regime -- Source segment
    regime0 -- First segment
    regime1 -- Second segment
    """
    if regime0.num_segments <= 1 and regime1.num_segments <= 1:
        return

    # Default pruning
    per = ws.segment_sample_ratio
    _removeNoise_aux(regime0, regime1, per)
    costC = _scanMinDiff(current_regime, regime0, regime1)

    # Optimal segment set
    opt0 = _getS("", "")
    opt1 = _getS("", "")
    regime0.copy_to(opt0)
    regime1.copy_to(opt1)
    prev = VERY_LARGE_COST

    # Find optimal pruning point
    while per <= ws.segment_sample_ratio * 10:
        if costC >= VERY_LARGE_COST:
            break

        per *= 2
        _removeNoise_aux(regime0, regime1, per)

        if regime0.num_segments <= 1 or regime1.num_segments <= 1:
            break

        costC = _scanMinDiff(current_regime, regime0, regime1)

        if prev > costC:
            regime0.copy_to(opt0)
            regime1.copy_to(opt1)
        else:
            break

        prev = costC

    opt0.copy_to(regime0)
    opt1.copy_to(regime1)
    ws.S.append(opt0)
    ws.S.append(opt1)


def _cps(current_regime, regime0, regime1, RM):
    """
    Cut point search with optional noise removal.

    Parameters:
    current_regime -- Source segment
    regime0 -- First segment
    regime1 -- Second segment
    RM -- Flag for noise removal
    """
    CPSearch(current_regime, regime0, regime1, ws)

    if RM:
        _removeNoise(current_regime, regime0, regime1)

    regime0.compute_likelihood_mdl()
    regime1.compute_likelihood_mdl()


def _findCentroid(current_regime, regime0, regime1, nsamples, seedlen):
    """
    Find optimal centroids for segmentation.

    Parameters:
    current_regime -- Source segment
    regime0 -- First output segment
    regime1 -- Second output segment
    nsamples -- Number of samples
    seedlen -- Seed length

    Returns:
    Minimum cost
    """
    costMin = VERY_LARGE_COST

    # Keep best seeds
    regime0stB, regime1stB, regime0lenB, regime1lenB = 0, 0, 0, 0  # Best
    regime0stC, regime1stC, regime0lenC, regime1lenC = 0, 0, 0, 0  # Current

    # Make sample set
    UniformSet(current_regime, seedlen, nsamples, ws.U)

    # Start uniform sampling
    for iter1 in range(ws.U.num_segments):
        for iter2 in range(iter1 + 1, ws.U.num_segments):
            UniformSampling(regime0, regime1, seedlen, iter1, iter2, ws.U)

            if regime0.num_segments == 0 or regime1.num_segments == 0:
                continue  # Not sufficient

            # Copy positions
            regime0stC = regime0.segments[0]['start']
            regime0lenC = regime0.segments[0]['duration']
            regime1stC = regime1.segments[0]['start']
            regime1lenC = regime1.segments[0]['duration']

            # Estimate HMM
            regime0.estimate_hmm_k(regime0.min_k)
            regime1.estimate_hmm_k(regime1.min_k)

            # Cut point search
            _cps(current_regime, regime0, regime1, True)

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
        FixedSampling(current_regime, regime0, regime1, seedlen)
        return VERY_LARGE_COST

    regime0.reset()
    regime1.reset()
    regime0.add_segment(regime0stB, regime0lenB)
    regime1.add_segment(regime1stB, regime1lenB)

    return costMin

def _regimeEst_aux(current_regime, regime0, regime1):
    """
    Estimate regimes from segments.

    Parameters:
    current_regime -- Source segment
    regime0 -- First output segment
    regime1 -- Second output segment
    """
    opt0 = _getS("", "")
    opt1 = _getS("", "")

    for i in range(ws.max_infer_iter):
        # Phase 1: Estimate parameters
        regime0.largest_segment()
        regime1.largest_segment()
        regime0.estimate_parameters()
        regime1.estimate_parameters()

        # Phase 2: Find cut-points
        _cps(current_regime, regime0, regime1, True)

        if regime0.num_segments == 0 or regime1.num_segments == 0:
            break  # Avoid null inference

        # If improving, update the optimal segment set
        diff = (opt0.costT + opt1.costT) - (regime0.costT + regime1.costT)
        if diff > 0:
            regime0.copy_to(opt0)
            regime1.copy_to(opt1)
        # If not improving, then break iteration (efficient convergence)
        elif i >= ws.min_infer_iter:
            break

    opt0.copy_to(regime0)
    opt1.copy_to(regime1)
    ws.S.append(opt0)
    ws.S.append(opt1)

def _regimeSplit(current_regime, regime0, regime1):
    """
    Split a regime into two.

    Parameters:
    current_regime -- Source segment
    regime0 -- First output segment
    regime1 -- Second output segment
    """
    seedlen = int(math.ceil(ws.lmax * ws.samplimg_lm))
    # Initialize HMM parameters
    _findCentroid(current_regime, regime0, regime1, ws.num_samples, seedlen)
    _regimeEst_aux(current_regime, regime0, regime1)

    if regime0.num_segments == 0 or regime1.num_segments == 0:
        return

    # Final model estimation
    regime0.estimate_parameters()
    regime1.estimate_parameters()

def _MDLtotal(Opt, C, verbose):
    """
    Calculate total MDL cost.

    Parameters:
    Opt -- Stack of optimal segments
    C -- Stack of candidate segments

    Returns:
    Total MDL cost
    """

    r = len(Opt) + len(C)
    m = sum(r.num_segments for r in Opt) + sum(r.num_segments for r in C)
    cost = sum(r.costT for r in Opt) + sum(r.costT for r in C)
    costT = cost + log_s(r) + log_s(m) + m * log_2(r) + (4 * 8) * r * r  # FLOATING_POINT_CONSTANT*r*r

    if verbose: sys.stdout.write(f"{r} {m} {costT:.0f} \n")

    return costT

def _viterbi(phmm, delta, st, length, vit):
    """
    Run Viterbi algorithm.

    Parameters:
    phmm -- HMM model
    delta -- Delta parameter
    st -- Start position
    length -- Segment length
    vit -- Viterbi data structure

    Returns:
    Coding cost
    """
    Lh = ViterbiL(phmm, length, ws.x.O[st:st + length], vit)

    if delta <= 0 or delta >= 1:
        raise ValueError('Delta must be 0.0 < delta < 1.0')

    Lh += math.log(delta)  # Switch
    Lh += (length - 1) * math.log(1.0 - delta)  # Else (stay)
    costC = -Lh / math.log(2.0)

    return costC


def _copyVitPath(q, st, length, vit):
    """
    Copy Viterbi path.

    Parameters:
    q -- Destination path array
    st -- Start position
    length -- Segment length
    vit -- Viterbi data structure
    """
    for i in range(length):
        q[i + st] = vit.q[i]

def FixedSampling(current_regime, regime0, regime1, length):
    """
    Fixed sampling strategy.

    Parameters:
    current_regime -- Source SegBox
    regime0 -- First output SegBox
    regime1 -- Second output SegBox
    length -- Length of samples
    """
    # Initialize segments
    regime0.reset()
    regime1.reset()

    # Segment regime0
    loc = 0 % current_regime.num_segments
    r = current_regime.segments[loc]['start']
    regime0.add_segment(r, length)

    # Segment regime1
    loc = 1 % current_regime.num_segments
    r = current_regime.segments[loc]['start'] + int(current_regime.segments[loc]['duration'] / 2)
    regime1.add_segment(r, length)

def UniformSet(current_regime, length, trial, U):
    """
    Uniform set strategy.

    Parameters:
    current_regime -- Source SegBox
    length -- Length of samples
    trial -- Number of trials
    U -- Output SegBox for uniform set
    """
    slideW = int(math.ceil((current_regime.total_length - length) / trial))

    # Create uniform blocks
    U.reset()

    for i in range(current_regime.num_segments):
        if U.num_segments >= trial:
            return

        st = current_regime.segments[i]['start']
        ed = st + current_regime.segments[i]['duration']

        for j in range(trial):
            next_pos = st + j * slideW

            if next_pos + length > ed:
                st = ed - length
                if st < 0:
                    st = 0
                U.add_segment_with_overlap(st, length)
                break

            U.add_segment_with_overlap(next_pos, length)


def UniformSampling(regime0, regime1, length, n1, n2, U):
    """
    Uniform sampling strategy.

    Parameters:
    current_regime -- Source SegBox
    regime0 -- First output SegBox
    regime1 -- Second output SegBox
    length -- Length of samples
    n1 -- First sample index
    n2 -- Second sample index
    U -- Uniform set SegBox
    """
    # Initialize segments
    regime0.reset()
    regime1.reset()

    i = int(n1 % U.num_segments)
    j = int(n2 % U.num_segments)

    st0 = U.segments[i]['start']
    st1 = U.segments[j]['start']

    # If overlapped, then ignore
    if abs(st0 - st1) < length:
        return

    regime0.add_segment(st0, length)
    regime1.add_segment(st1, length)


# Function declarations from cps.c
def _findMax(P, k):
    """
    Find the index of the maximum value in array P.

    Parameters:
    P -- Array of values
    k -- Length of array

    Returns:
    Index of maximum value
    """
    loc = -1
    max_val = float('-inf')
    for i in range(k):
        if max_val < P[i]:
            max_val = P[i]
            loc = i
    return loc


def _reset_npaths(nS, k):
    """
    Reset path counts.

    Parameters:
    nS -- Array of path counts
    k -- Length of array
    """
    for i in range(k):
        nS[i] = 0


def _copy_path(from_path, nfrom, to_path):
    """
    Copy path array.

    Parameters:
    from_path -- Source path array
    nfrom -- Number of elements to copy
    to_path -- Destination path array

    Returns:
    Number of elements copied
    """
    for i in range(nfrom):
        to_path[i] = from_path[i]
    return nfrom


def _search_aux(st, length, regime0, regime1):
    """
    Auxiliary function for Cut Point Search.

    Parameters:
        st (int): Start index of the segment
        length (int): Length of the segment
        regime0 (Regime): First regime model
        regime1 (Regime): Second regime model

    Returns:
        float: Coding cost
    """
    global ws

    # Extract HMMs and deltas
    m0, d0 = regime0.model, regime0.delta
    m1, d1 = regime1.model, regime1.delta
    k0, k1 = m0.k, m1.k
    O = ws.x.O
    O_window = O[st:st + length]

    # Handle degenerate case early
    if d0 <= 0 or d1 <= 0:
        raise ValueError('Degenerate dlta <= 0')

    # Alias CPS structures
    Pu, Pv = ws.cps.Pu, ws.cps.Pv
    Pi, Pj = ws.cps.Pi, ws.cps.Pj
    Su, Sv = ws.cps.Su, ws.cps.Sv
    Si, Sj = ws.cps.Si, ws.cps.Sj
    nSu, nSv = ws.cps.nSu, ws.cps.nSv
    nSi, nSj = ws.cps.nSi, ws.cps.nSj

    # Reset path state
    _reset_npaths(nSu, k0)
    _reset_npaths(nSv, k0)
    _reset_npaths(nSi, k1)
    _reset_npaths(nSj, k1)

    # Precompute emissions
    log_emission0 = batch_log_pdf(m0, k0, length, O_window)  # shape: (k0, length)
    log_emission1 = batch_log_pdf(m1, k1, length, O_window)  # shape: (k1, length)

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
    for t in range(st + 1, st + length):
        offset = t - st

        # --- Update Pu[t] ---

        # Find best path from previous Pj
        maxj = _findMax(Pj, k1)

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
                nSu[u] = _copy_path(Sj[maxj], nSj[maxj], Su[u])
                Su[u][nSu[u]] = t
                nSu[u] += 1
            else:
                Pu[u] = stay_score
                nSu[u] = _copy_path(Sv[maxv], nSv[maxv], Su[u])

        # --- Update Pi[t] ---

        maxv = _findMax(Pv, k0)

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
                nSi[i] = _copy_path(Sv[maxv], nSv[maxv], Si[i])
                Si[i][nSi[i]] = t
                nSi[i] += 1
            else:
                Pi[i] = stay_score
                nSi[i] = _copy_path(Sj[maxj], nSj[maxj], Si[i])

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
    curSt = st
    flip = firstID  # Either S0 or S1 depending on best regime

    for i in range(npath):
        nxtSt = path[i]
        target_regime = regime0 if flip == S0 else regime1
        target_regime.add_segment(curSt, nxtSt - curSt)
        curSt = nxtSt
        flip *= -1  # Alternate regime

    # Final segment
    target_regime = regime0 if flip == S0 else regime1
    target_regime.add_segment(curSt, st + length - curSt)

    # Compute and return coding cost in bits
    costC = -lh / math.log(2.0)
    return costC


def CPSearch(current_regime, regime0, regime1, wsd):
    """
    Cut Point Search algorithm.

    Parameters:
    current_regime -- Input segment box
    regime0 -- First output segment box
    regime1 -- Second output segment box
    wsd -- AutoPlait workspace

    Returns:
    Likelihood (coding cost)
    """
    global ws
    ws = wsd

    regime0.reset()
    regime1.reset()

    lh = 0
    for i in range(current_regime.num_segments):
        lh += _search_aux(current_regime.segments[i]['start'], current_regime.segments[i]['duration'], regime0, regime1)

    return lh

# ------------------------------
#         hmm_header_py.py
# ------------------------------
class HMM:
    """
    Hidden Markov Model class.

    Attributes:
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

    def reset(self, K, D):
        """Reset an HMM to random values."""
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

    def copy_to(self, phmm2):
        """Copy HMM phmm1 to phmm2."""
        phmm2.d = self.d
        phmm2.k = self.k
        phmm2.n = self.n

        # Vectorized matrix/array copies
        phmm2.A[:self.k, :self.k] = self.A[:self.k, :self.k]
        phmm2.mean[:self.k, :self.d] = self.mean[:self.k, :self.d]
        phmm2.var[:self.k, :self.d] = self.var[:self.k, :self.d]
        phmm2.sum_w[:self.k, :self.d] = self.sum_w[:self.k, :self.d]
        phmm2.M2[:self.k, :self.d] = self.M2[:self.k, :self.d]
        phmm2.pi[:self.k] = self.pi[:self.k]
        phmm2.A_denom[:self.k] = self.A_denom[:self.k]

        # Scalar copy
        phmm2.pi_denom = self.pi_denom


class BAUM:
    """
    Baum-Welch algorithm data class.

    Attributes:
        alpha: Forward probabilities [mx][k]
        beta: Backward probabilities [mx][k]
        gamma: State probabilities [n][mx][k]
        gamma_space: Space for gamma calculations
        xi: Transition probabilities [n][mx][k][k]
        scale: Scaling factors [mx]
        idx: Cluster indices for k-means [n][mx]
        chmm: Current HMM model
    """

    def __init__(self, variance, delta, max_iter, k_means_max_iter):

        self.default_variance = variance
        self.delta = delta
        self.max_iter = max_iter
        self.k_means_max_iter = k_means_max_iter

        self.alpha = None
        self.beta = None
        self.gamma = None
        self.gamma_space = None
        self.xi = None
        self.scale = None
        self.idx = None
        self.chmm = HMM(self.default_variance)


class VITERBI:
    """
    Viterbi algorithm data class.

    Attributes:
        delta: Viterbi probabilities [m][k]
        psi: Backtracking indices [m][k]
        q: State sequence [m]
        piL: Log initial probabilities [k]
        AL: Log transition probabilities [k][k]
        biot: Log emission probabilities [k][m]
    """

    def __init__(self):
        self.delta = None
        self.psi = None
        self.q = None
        self.piL = None
        self.AL = None
        self.biot = None

def PrintHMM(fp, phmm):
    """Print HMM parameters to a file pointer."""
    fp.write(f"k= {phmm.k}\n")
    fp.write(f"d= {phmm.d}\n")

    fp.write("pi:\n")
    for i in range(phmm.k):
        fp.write(f"{phmm.pi[i]} ")
    fp.write("\n")

    fp.write("A:\n")
    for i in range(phmm.k):
        for j in range(phmm.k):
            fp.write(f"{phmm.A[i][j]} ")
        fp.write("\n")

    fp.write("mean:\n")
    for j in range(phmm.k):
        for k in range(phmm.d):
            fp.write(f"{phmm.mean[j][k]} ")
        fp.write("\n")

    fp.write("var:\n")
    for j in range(phmm.k):
        for k in range(phmm.d):
            fp.write(f"{phmm.var[j][k]} ")
        fp.write("\n")


def InitHMM(phmm, K, D):
    """Initialize an HMM with K states and D dimensions."""
    phmm.n = 0
    phmm.d = D
    phmm.k = K

    # Allocate arrays
    phmm.A = np.zeros((phmm.k, phmm.k), dtype=np.float64)
    phmm.pi = np.zeros(phmm.k, dtype=np.float64)
    phmm.mean = np.zeros((phmm.k, phmm.d), dtype=np.float64)
    phmm.var = np.zeros((phmm.k, phmm.d), dtype=np.float64)
    phmm.pi_denom = 0
    phmm.A_denom = np.zeros(phmm.k, dtype=np.float64)
    phmm.sum_w = np.zeros((phmm.k, phmm.d), dtype=np.float64)
    phmm.M2 = np.zeros((phmm.k, phmm.d), dtype=np.float64)

    # Random A matrix with EPSILON added, normalized row-wise
    phmm.A = np.random.rand(K, K) + EPSILON
    phmm.A /= phmm.A.sum(axis=1, keepdims=True)

    # Random pi with EPSILON added, normalized
    phmm.pi = np.random.rand(K) + EPSILON
    phmm.pi /= phmm.pi.sum()

    # Random means, fixed variance
    phmm.mean = np.random.rand(K, D) * MAX_NORMALISATION_VALUE
    phmm.var.fill(phmm.default_variance)

def batch_log_pdf(hmm, k, m, observations):
    observations = observations[:m]  # (T, D)
    mean = hmm.mean[:k] # (K, D)
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

# Function declarations from forbackward.c
def forward(phmm, O, m, alpha, scale, emission_probs):
    """
    Run the Forward algorithm to calculate the likelihood of observations.

    Parameters:
    phmm -- HMM model
    O -- observation sequence
    m -- sequence length
    alpha -- forward probabilities matrix
    scale -- scaling factors array

    Returns:
    L -- log likelihood of the observation sequence
    """
    # Initialize scale
    scale[:m] = 0.0

    O0 = np.asarray([O[0]])  # shape: (1, D)
    obs_probs = batch_pdf(phmm, phmm.k, m, O0)[:, 0]  # shape: (K,)
    alpha[0, :phmm.k] = phmm.pi[:phmm.k] * obs_probs
    scale[0] = alpha[0].sum()

    alpha[0, :phmm.k] /= scale[0]

    # Induction
    for t in range(m - 1):
        # Matrix multiply: alpha[t] (1, K) x A (K, K) -> (K,)
        alpha_t1 = alpha[t, :phmm.k] @ phmm.A[:phmm.k, :phmm.k]  # shape: (K,)
        # Multiply by emission probabilities at time t+1
        alpha[t + 1, :phmm.k] = alpha_t1 * emission_probs[:phmm.k, t + 1]

        # Normalize and store scale
        scale[t + 1] = alpha[t + 1, :phmm.k].sum()
        alpha[t + 1, :phmm.k] /= scale[t + 1]

    # Termination
    return np.sum(np.log(scale[:m]))


def backward(phmm, m, beta, scale, emission_probs):
    """
    Run the Backward algorithm.

    Parameters:
    phmm -- HMM model
    O -- observation sequence
    m -- sequence length
    beta -- backward probabilities matrix
    scale -- scaling factors array from Forward algorithm

    Returns:
    L -- log likelihood of the observation sequence
    """

    # Init (t=m-1)
    beta[m - 1, :phmm.k] = 1.0 / scale[m - 1]

    # Induction
    for t in range(m - 2, -1, -1):
        # Emission at time t+1 across all states (shape: K,)
        emis = emission_probs[:phmm.k, t + 1]

        # Elementwise multiply: emission * beta[t+1]
        weighted = emis * beta[t + 1, :phmm.k]  # shape: (K,)

        # Matrix multiply: A (KxK) x weighted (K,) → (K,)
        beta[t, :phmm.k] = phmm.A[:phmm.k, :phmm.k] @ weighted

        # Normalize by scale[t]
        beta[t] /= scale[t]

    # Log likelihood (same as Forward)
    return np.sum(np.log(scale[:m]))


# Function declarations from baum.c
def _computeGamma(phmm, alpha, beta, gamma, m):
    """
    Compute gamma values (state probabilities).

    Parameters:
    phmm -- HMM model
    alpha -- forward probabilities matrix
    beta -- backward probabilities matrix
    gamma -- output gamma matrix
    m -- sequence length
    """
    product = alpha[:m, :phmm.k] * beta[:m, :phmm.k]  # (m, K)
    gamma[:m, :phmm.k] = product / product.sum(axis=1, keepdims=True)


def _computeXi(phmm, alpha, beta, xi, m, emission_probs):
    """
    Compute xi values (transition probabilities).

    Parameters:
    phmm -- HMM model
    O -- observation sequence
    alpha -- forward probabilities matrix
    beta -- backward probabilities matrix
    xi -- output xi tensor
    m -- sequence length
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
        outer = alpha[t, :k][:, None] * A * emission_probs[:k, t + 1][None, :] * beta[t + 1, :k][None, :]

        # Normalize
        xi[t] = outer / np.sum(outer)

def _computeParams(phmm, gamma, xi, xlst):
    """
    Compute HMM parameters based on gamma and xi values.

    Parameters:
    phmm -- HMM model to update
    gamma -- state probabilities
    xi -- transition probabilities
    xlst -- observation sequences
    """
    k = phmm.k
    # Initial probability vector pi
    # (a) recover previous pi*N
    phmm.pi[:k] *= phmm.pi_denom  # restore weighted prior

    # (b) add new gamma
    gamma0_sum = sum(g[0] for g in gamma[:k])  # g[0] is shape (K,)
    phmm.pi[:k] += EPSILON + gamma0_sum[:k]

    # # (c) normalize, clip negatives and normalize
    np.clip(phmm.pi[:k], 0, None, out=phmm.pi[:k])
    phmm.pi_denom = np.sum(phmm.pi[:k])

    phmm.pi[:k] /= phmm.pi_denom

    # Transition matrix A
    # (a) recover previous A*N
    phmm.A[:k, :k] *= phmm.A_denom[:k, None]  # (K, 1)

    # (b) add new xi
    # (b) accumulate all xi transitions
    A_update = np.zeros((k, k), dtype=np.float64)
    for r in range(phmm.n):
        m_r = xlst[r].m
        A_update += np.sum(xi[r][:m_r - 1], axis=0) + EPSILON  # sum over t

    phmm.A[:k, :k] += A_update

    np.clip(phmm.A, 0.0, None, out=phmm.A)
    phmm.A_denom = np.sum(phmm.A, axis=1)
    phmm.A /= phmm.A_denom[:, None]

    # Weighted incremental computation for means and variances
    for i in range(k):
        for j in range(phmm.d):
            mean = phmm.mean[i][j]
            M2 = phmm.M2[i][j]
            sum_w = phmm.sum_w[i][j]

            if sum_w == 0.0:
                mean = 0.0
                M2 = 0.0

            for r in range(phmm.n):
                obs = xlst[r].O[:, j]  # shape: (T,)
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
            var = min(ws.max_variance, max(var, ws.min_variance))

            phmm.mean[i][j] = mean
            phmm.M2[i][j] = M2
            phmm.sum_w[i][j] = sum_w
            phmm.var[i][j] = var


def BaumWelch(phmm, n, xlst, baum, wantKMEANS):
    """
    Perform the Baum-Welch algorithm for HMM training.

    Parameters:
    phmm -- HMM model to train
    n -- number of sequences
    xlst -- observation sequences
    baum -- BAUM data structure
    wantKMEANS -- flag for K-means initialization

    Returns:
    Lsum -- final log likelihood
    """

    l = 0
    alpha = baum.alpha
    beta = baum.beta
    gamma = baum.gamma
    xi = baum.xi
    scale = baum.scale
    idx = baum.idx

    if n == 0:
        raise ValueError('Estimation error (n == 0)')

    # If k==1, nothing to do (just run K-means)
    if phmm.k == 1:
        Kmeans(phmm, n, xlst, phmm.d, phmm.k, idx, baum.k_means_max_iter)
        return -1

    if wantKMEANS:
        # If initial stage and want k-means
        if phmm.n == 0:
            phmm.reset(phmm.k, phmm.d)
            Kmeans(phmm, n, xlst, phmm.d, phmm.k, idx, baum.k_means_max_iter)

    # Take absolute value of n
    n = int(abs(float(n)))

    prev_n = phmm.n
    phmm.n = n
    phmm.copy_to(baum.chmm)

    # For each sequence
    Lsum = 0.0
    for r in range(phmm.n):
        O = xlst[r].O
        m = xlst[r].m

        # Cache emission probs for this sequence
        emission_probs = batch_pdf(phmm, phmm.k, m, O)  # shape: (K, m)

        Lf = forward(phmm, O, m, alpha, scale, emission_probs)
        backward(phmm, m, beta, scale, emission_probs)
        _computeGamma(phmm, alpha, beta, gamma[r], m)
        _computeXi(phmm, alpha, beta, xi[r], m, emission_probs)
        Lsum += Lf

    # Log likelihood
    Lpreb = Lsum

    # Baum-Welch iterations
    while True:
        # M-STEP: Update model parameters
        baum.chmm.copy_to(phmm)
        _computeParams(phmm, gamma, xi, xlst)

        # E-STEP: Re-estimate expectations
        Lsum = 0.0
        for r in range(phmm.n):
            O = xlst[r].O
            m = xlst[r].m

            # Cache emission probs for this sequence
            emission_probs = batch_pdf(phmm, phmm.k, m, O)  # shape: (K, m)

            Lf = forward(phmm, O, m, alpha, scale, emission_probs)
            backward(phmm, m, beta, scale, emission_probs)
            _computeGamma(phmm, alpha, beta, gamma[r], m)
            _computeXi(phmm, alpha, beta, xi[r], m, emission_probs)
            Lsum += Lf

        delta = Lpreb - Lsum
        Lpreb = Lsum
        l += 1

        # Check convergence
        if abs(delta) <= baum.delta or l >= baum.max_iter:
            break

    # For incremental fitting
    _computeParams(phmm, gamma, xi, xlst)

    # Avoid numerical errors
    if math.isnan(Lsum):
        print("baumWelch: isnan, resetHMM...")
        phmm.reset(phmm.k, phmm.d)

    phmm.n = phmm.n + prev_n

    return Lsum


def AllocBaum(baum, n, m, k, d):
    """
    Allocate memory for Baum-Welch algorithm data structures.

    Parameters:
    baum -- BAUM data structure
    n -- number of sequences
    m -- max sequence length
    k -- number of states
    d -- dimension of observations
    """

    baum.alpha = np.zeros((m, k), dtype=np.float64)#dmatrix(0, m, 0, k)
    baum.beta = np.zeros((m, k), dtype=np.float64)#dmatrix(0, m, 0, k)
    baum.scale = np.zeros(m, dtype=np.float64)#dvector(0, m)
    baum.idx = np.zeros((n, m), dtype=np.int32)#imatrix(0, n, 0, m)

    # Gamma
    baum.gamma = [None] * n
    for r in range(n):
        baum.gamma[r] = np.zeros((m, k), dtype=np.float64)#dmatrix(0, m, 0, k)

    # Xi
    baum.xi = [None] * n
    for r in range(n):
        baum.xi[r] = [None] * m
        for t in range(m):
            baum.xi[r][t] = np.zeros((k, k), dtype=np.float64)#dmatrix(0, k, 0, k)

    # For incremental EM
    InitHMM(baum.chmm, k, d)


# Function declarations from viterbi.c
def AllocViterbi(vit, n, m, k):
    """Allocate memory for Viterbi algorithm data structures."""
    vit.delta = np.zeros((m, k), dtype=np.float64)
    vit.psi = np.zeros((m, k), dtype=np.int32)
    vit.q = np.zeros(m, dtype=np.int32)
    vit.piL = np.zeros(k, dtype=np.float64)
    vit.AL = np.zeros((k, k), dtype=np.float64)
    vit.biot = np.zeros((k, m), dtype=np.float64)

def TakelogHMM(phmm, vit, m, O):
    """
    Take logarithm of HMM parameters for log-domain Viterbi algorithm.

    Parameters:
    phmm -- HMM model
    vit -- VITERBI data structure
    m -- sequence length
    O -- observation sequence
    """

    vit.piL[:phmm.k] = np.log(phmm.pi[:phmm.k] + EPSILON)

    #vit.AL[:phmm.k, :phmm.k] = np.log(phmm.A[:phmm.k, phmm.k] + EPSILON)
    for i in range(phmm.k):
        for j in range(phmm.k):
            vit.AL[i][j] = math.log(phmm.A[i][j] + EPSILON)

    m = min(m, O.shape[0])
    vit.biot[:phmm.k, :m] = batch_log_pdf(phmm, phmm.k, m, O[:m])

def ViterbiL(phmm, m, O, vit):
    """
    Run the Viterbi algorithm in log domain.

    Parameters:
    phmm -- HMM model
    m -- sequence length
    O -- observation sequence
    vit -- VITERBI data structure

    Returns:
    Lh -- log likelihood of the most probable path
    """

    k = phmm.k

    if m == 0:
        return 0

    TakelogHMM(phmm, vit, m, O)

    delta = vit.delta
    psi = vit.psi
    q = vit.q
    piL = vit.piL
    AL = vit.AL
    biot = vit.biot

    # compute delta (t==0)
    delta[0, :k] = piL[:k] + biot[:k, 0]
    psi[0, :k] = 0

    # compute delta (t>0)
    for t in range(1, m):
        deltax = -VERY_LARGE_COST
        for j in range(phmm.k):
            maxval = -VERY_LARGE_COST
            maxvalind = 0
            for i in range(phmm.k):
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
    q[m - 1] = -1
    for i in range(phmm.k):
        if delta[m - 1][i] > Lh:
            Lh = delta[m - 1][i]
            q[m - 1] = i

    # avoid error
    if q[m - 1] == -1:
        raise ValueError("Cannot compute log Viterbi path")

    # check viterbi path
    for t in range(m - 2, -1, -1):
        q[t] = psi[t + 1][q[t + 1]]

    return Lh

def compute_cluster_stats(data, labels, k, d):
    """
    Compute per-cluster mean and variance (vectorized).
    Parameters:
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
            vars_[j] = ws.min_variance
            continue

        mean = np.mean(cluster_points, axis=0)
        var = np.var(cluster_points, axis=0, ddof=1)  # sample variance

        means[j] = mean
        vars_[j] = np.clip(var + EPSILON, ws.min_variance, ws.max_variance)

    return means, vars_

def Kmeans(phmm, n, xlst, d, k, idx, max_iter):
    """Perform K-means clustering.

    Parameters:
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
        X = np.vstack([x.O for x in xlst[:n]])

        # Assign all to cluster 0
        for r in range(n):
            idx[r][:xlst[r].m] = 0

        # Compute global mean and variance
        phmm.mean[0] = np.mean(X, axis=0)
        var = np.var(X, axis=0, ddof=1)
        phmm.var[0] = np.clip(var + EPSILON, ws.min_variance, ws.max_variance)

        return

    # Flatten data
    X = np.vstack([x.O for x in xlst[:n]])  # shape: (N_total, D)

    kmeans = KMeans(n_clusters=k, init='random', n_init=1, max_iter=max_iter, random_state=42)

    labels = kmeans.fit_predict(X)

    offset = 0
    for r in range(n):
        for i in range(xlst[r].m):
            idx[r][i] = labels[offset]
            offset += 1

    # Final calculation of means and variances
    phmm.mean[:k], phmm.var[:k] = compute_cluster_stats(X, labels, k, d)

# Function declarations from dynamic.c
def Split(X, st, len_val, x):
    """Split a sequence X into a sub-sequence x."""
    if st < 0:
        st = 0
    if X.m < st + len_val:
        len_val = X.m - st

    # In Python, we need to create a view of the array rather than pointer arithmetic
    x.O = X.O[st:st + len_val]
    x.m = len_val
    x.parent = X.id
    x.st = st
    x.tag = f"[{x.id}] {X.tag} [{st}-{st + len_val}]({len_val})"

# ------------------------------
#         MATH FUNCTIONS
# ------------------------------
# MDL (Minimum Description Length) functions
def log_2(x):
    """Log base 2 of x."""
    return math.log(x) / math.log(2.0)

def log_s(x):
    """Compute 2*log_2(x) + 1."""
    return 2.0 * log_2(x) + 1.0
