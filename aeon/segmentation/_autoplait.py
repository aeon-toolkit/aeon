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
        t_params, f = [], []
        d = []

        q = []  # Stack
        s = []  # Output segment set
        m = 0  # Output number of segments
        r = 0  # Output number of regimes
        s.append((1, len(X)))  # 1 initial segment; the entire TS
        m_0 = 1  # There is 1 initial segment

        t0 = []  # TODO: Estimate t0 of s_0
        q.append((m_0, s[0], t0))  # Push onto stack
        while q:  # While the stack is not empty
            m_i, s_i, t_i = q.pop()
            m1, m2, s1, s2, t1, t2, d = _regime_split(X)
            if _cost(X, t1, t2, d) < _cost(X, t_i):
                q.append((m1, s1, t1))
                q.append((m2, s2, t2))
            else:
                s.append(s_i)
                t_params.append(t_i)
                r += 1
                # TODO: Update regime transitions d
                # TODO: segment membership f
                m += m_i
        t_params.append(d)  # Add the regime transition matrix
        results = {
            "num_segments":m,
            "num_regimes":r,
            "segment_set":s,
            "regime_parameters":t_params,
            "segment_memberships":f,
        }
        return results

    def complete_parameters(self, as_list:bool = False):
        if as_list:
            return list(self._complete_parameters.values())
        else:
            return self._complete_parameters


class AutoPlaitSegmenter(BaseSegmenter):
    """
    DOCSTRING
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
        pass
    else:
        # TODO: Cost of single regime
        pass
