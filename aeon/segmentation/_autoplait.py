"""AutoPlait Segmentation."""

__maintainer__ = []
__all__ = ["AutoPlaitSegmenter"]

import numpy as np

from aeon.segmentation.base import BaseSegmenter


class _AutoPlait:

    _complete_parameters = {}

    def __init__(self):
        pass

    def predict(self, X):
        self._complete_parameters = self._autoplait(X)

    def _autoplait(self, X):
        regime_parameters, segment_membership = [], []
        transition_matrix = []

        regime_stack = []  # Stack
        segment_set = []  # Output segment set
        num_segments = 0  # Output number of segments
        num_regimes = 0  # Output number of regimes
        segment_set.append((1, len(X)))  # 1 initial segment; the entire TS
        num_segments_0 = 1  # There is 1 initial segment

        t0 = []  # TODO: Estimate t0 of s_0
        regime_stack.append((num_segments_0, segment_set[0], t0))  # Push onto stack
        while regime_stack:  # While the stack is not empty
            current_num_segments, current_segment_set, t_i = regime_stack.pop()
            m1, m2, s1, s2, t1, t2, transition_matrix = _regime_split(X)
            if _cost(X, t1, t2, transition_matrix) < _cost(X, t_i):
                regime_stack.append((m1, s1, t1))
                regime_stack.append((m2, s2, t2))
            else:
                segment_set.append(current_segment_set)
                regime_parameters.append(t_i)
                num_regimes += 1
                # TODO: Update regime transitions d
                # TODO: segment membership f
                num_segments += current_num_segments
        regime_parameters.append(transition_matrix)  # Add the regime transition matrix
        results = {
            "num_segments":num_segments,
            "num_regimes":num_regimes,
            "segment_set":segment_set,
            "regime_parameters":regime_parameters,
            "segment_memberships":segment_membership,
        }
        return results

    def complete_parameters(self, as_list:bool = False):
        if as_list:
            return list(self._complete_parameters.values())
        else:
            return self._complete_parameters


class AutoPlaitSegmenter(BaseSegmenter):
    """AutoPlait Segmentation.

    Using ClaSP [1]_, [2]_ for the CPD problem is straightforward: We first compute the
    profile and then choose its global maximum as the change point. The following CPDs
    are obtained using a bespoke recursive split segmentation algorithm.

    Parameters
    ----------
    param_name : param_type, default = NONE
        Param desc.

    References
    ----------
    .. [1] AUTOPLAIT REFERENCE.
    .. [2] Schafer, Patrick and Ermshaus, Arik and Leser, Ulf. "ClaSP - Time Series
    Segmentation", CIKM, 2021.

    Examples
    --------
    >>> from aeon.segmentation import AutoPlaitSegmenter
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, _, cps = load_gun_point_segmentation()
    >>> autoplait = AutoPlaitSegmenter()
    >>> found_cps = autoplait.fit_predict(X)
    """

    _tags = {
        "returns_dense": True,
        "fit_is_empty": True,
    }

    _X = None
    _autoplait_results = None

    def __init__(self):
        self._autoplait = _AutoPlait()
        super().__init__(axis=0)

    def _predict(self, X) -> np.ndarray:
        """Perform AutoPlait segmentation.

        Parameters
        ----------
        X: np.ndarray
            2D time series shape (n_timepoints, n_channels).

        Returns
        -------
        y_pred : np.ndarray
            DESC
        """
        self._autoplait.predict(X)
        self.is_fitted = True
        params = self._autoplait.complete_parameters()
        cut_points = [e[1] for e in params["segment_set"]]
        return np.array(cut_points)

    def complete_parameters(self, as_list:bool = False):
        self._check_is_fitted()
        return self._autoplait.complete_parameters(as_list)

def _cut_point_search(X, regime1, regime2, transition_matrix):
    m1, m2 = 0, 0
    s1, s2 = [], []
    l1, l2 = [], [] # Candidate cut point sets
    for t in range(len(X)):
        # TODO: Compute likelihoods for all states of regime1 and regime2
        # TODO: Update candidate cut point sets for both regimes
        pass
    l_best = best_cut_point_set(l1, l2)
    t_s = 1
    for idx, l_i in enumerate(l_best):
        s_i = (t_s, l_i)
        if idx % 2 == 1:
            s1.append(s_i)
            m1 += 1
        else:
            s2.append(s_i)
            m2 += 1
        t_s = l_i
    return m1, m2, s1, s2

def best_cut_point_set(cut_set1, cut_set2):
    # TODO: best_cut_point_set(l1, l2)
    return cut_set1

def _regime_split(X):
    m1, m2, s1, s2 = 0, 0, [], [] # Initialise outputs
    t1, t2, d = [], [], [] # TODO: Initialise models
    current_cost = float('inf') # Current cost is infinity
    while c := _cost(X, t1, t2, d) < current_cost:
        current_cost = c
        m1, m2, s1, s2 = _cut_point_search(X, t1, t2, d)
        # TODO: Update model parameters t1, t2
        # TODO: Update regime transitions d
    return m1, m2, s1, s2, t1, t2, d

def _cost(X, regime1, regime2=None, transition_matrix=None):
    if regime2 is not None:
        # TODO: Cost of regime pair
        return 0
    else:
        # TODO: Cost of single regime
        return 0

def _log_star(x):
    count = 0
    while x > 1:
        x = math.log2(x)  # Apply base-2 logarithm
        count += 1
    return count

def _new_cost(X, S, regimes):
    n, d = X.shape
    m = len(S)
    r = len(regimes[:-1])
    log_star_values = (_log_star(n), _log_star(d), _log_star(m), _log_star(r))
    log_star_sum = sum(log_star_values)

    segment_cost = _segment_length_cost(S)
    cost_m_value = _cost_m(regimes)
    cost_e_value = _cost_e(X, regimes)

    return log_star_sum + m * log_star_values[3] + segment_cost + cost_m_value + cost_e_value

def _segment_length_cost(S):
    total = 0
    for s in S:
        total += _log_star(abs(s[1] - s[0]))
    return total

def _cost_m(regimes):
    total = 0
    cf = 4 * 8
    d = 1
    for regime in regimes[:-1]:
        k = regime.hidden_states
        total += _log_star(k) + cf * (k + k**2 + 2*k*d)
    return total + cf * len(regimes[:-1])**2

def _cost_e(X, regime):
    return 1

def _score_segmentation(pred_regimes, true_regimes, n):
    sum_diff = 0
    num_regimes = len(true_regimes)
    for i in range(num_regimes):
        current = pred_regimes[i]
        closest = min(true_regimes, key=lambda x: abs(x - current))
        diff = abs(closest - current)
        sum_diff += diff
    return sum_diff/n

def equation_7(X, regime1:_HiddenMarkovModel, regime2:_HiddenMarkovModel, rtm):
    if rtm.shape != (2, 2):
        raise ValueError("Regime transition matrix shape must be (2,2)")
    max_reg1 = max([handle_regime_1(X, regime1, regime2, rtm, i, len(X)) for i in range(regime1.hidden_states)])
    max_reg2 = max([handle_regime_2(X, regime1, regime2, rtm, u, len(X)) for u in range(regime2.hidden_states)])
    return max(max_reg1, max_reg2)

def handle_regime_1(X, regime1:_HiddenMarkovModel, regime2:_HiddenMarkovModel, rtm, state, t):
    if t < 0:
        raise ValueError("Undefined for time t < 0")
    if t == 0:
        return rtm[0][0] * regime1.initial_probs[state] * regime1.output_probabilities[state](X[0])
