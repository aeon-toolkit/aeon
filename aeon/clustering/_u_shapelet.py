"""Unsupervised Shapelet Clustering Algorithm."""

from collections import defaultdict
from typing import Optional, Union

import numpy as np

from aeon.clustering.base import BaseClusterer


class UShapeletClusterer(BaseClusterer):
    """
    Unsupervised Shapelet Clustering Algorithm for univariate time series.

    This algorithm extracts shapelets from univariate time series data using a
    Symbolic Aggregate approXimation (SAX) representation and collision counting,
    and clusters the time series based on the distances to the best shapelet.

    Parameters
    ----------
    subseq_length : int, default=10
        Length of the subsequences to consider as candidate shapelets.
    sax_length : int, default=16
        Length of the PAA transformation used in the SAX representation.
    projections : int, default=10
        Number of random projections used in collision counting.
    lb : int, default=2
        Lower bound on the number of time series in a group when evaluating the
        gap score.
    ub : int, optional, default=None
        Upper bound on the number of time series in a group when evaluating the
        gap score.
        If None, it is set based on the number of time series.
    random_state : int or np.random.RandomState, optional, default=None
        Seed or random number generator for reproducibility.

    References
    ----------
    ..[1] Ulanova, L., Begum, N., & Keogh, E. (2015). Scalable Clustering of
      Time Series with U‐Shapelets. In Proceedings of the 2015 SIAM International
      Conference on Data Mining (SDM) (pp. 900–908).

    ..[2] Zakaria, J., Mueen, A., & Keogh, E. (2012). Clustering Time Series
      Using Unsupervised‐Shapelets. In 2012 IEEE 12th International Conference
      on Data Mining (ICDM) (pp. 785–794).

    ..[3] Zakaria, J., Mueen, A., Keogh, E., & Young, N. (2016). Accelerating the
      Discovery of Unsupervised‐Shapelets. Data Mining and Knowledge Discovery,
      30(2), 243–281.
    """

    _tags = {
        "capability:multivariate": False,
        "algorithm_type": "feature",
    }

    def __init__(
        self,
        subseq_length: int = 10,
        sax_length: int = 16,
        projections: int = 10,
        lb: int = 2,
        ub: int = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.subseq_length = subseq_length
        self.sax_length = sax_length
        self.projections = projections
        self.lb = lb
        self.ub = ub
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.best_shapelet_ = None
        self.best_gap_ = None
        self.best_split_ = None
        self.best_loc_ = None

        super().__init__()

    def _fit(self, X: np.ndarray, y=None) -> "UShapeletClusterer":
        """
        Fit the UShapeletClusterer to the provided time series data.

        This method extracts the best shapelet from the data by computing the gap score
        over candidate shapelets.

        Parameters
        ----------
        X : np.ndarray or list of arrays
            List of univariate time series.
        y : None
            Ignored.

        Returns
        -------
        self : UShapeletClusterer
            Fitted instance of UShapeletClusterer.
        """
        self.best_shapelet_, self.best_gap_, self.best_split_, self.best_loc_ = (
            self.get_best_u_shapelet(
                X,
                self.subseq_length,
                self.sax_length,
                self.projections,
                self.lb,
                self.ub,
            )
        )
        return self

    def _predict(self, X, y=None):
        """
        Predict cluster labels for each time series in X.

        The prediction is based on the distance of each time series to the best
        shapelet.Each series is assigned to cluster 0 if its distance is below a
        threshold determined by the split value, otherwise to cluster 1.

        Parameters
        ----------
        X : np.ndarray or list of arrays
            List of univariate time series.
        y : None
            Ignored.

        Returns
        -------
        labels : np.ndarray
            Array of cluster labels (0 or 1) for each time series.
        """
        distances = []
        L = len(self.best_shapelet_)
        for series in X:
            series = np.squeeze(np.array(series))
            min_dist = float("inf")
            for j in range(0, len(series) - L + 1):
                sub = self.z_normalize(series[j : j + L])
                d = np.linalg.norm(self.best_shapelet_ - sub)
                if d < min_dist:
                    min_dist = d
            distances.append(min_dist)
        threshold = sorted(distances)[self.best_split_ - 1]  # 0-indexed
        return np.array([0 if d <= threshold else 1 for d in distances])

    def z_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Z-normalize a time series (1D array).

        Parameters
        ----------
        X : np.ndarray
            Input time series array.

        Returns
        -------
        np.ndarray
            Z-normalized time series (zero mean and unit variance).
            If the standard deviation is below 1e-6, returns an array of zeros.
        """
        X = np.array(X, dtype=float)
        mean = X.mean()
        std = X.std()
        if std < 1e-6:
            return np.zeros_like(X)
        return (X - mean) / std

    def paa_transform(self, X: np.ndarray, n_segments: int):
        """
        Perform Piecewise Aggregate Approximation (PAA) transformation on a time series.

        Parameters
        ----------
        X : np.ndarray
            Input time series array.
        n_segments : int
            Number of segments to divide the time series into.

        Returns
        -------
        np.ndarray
            PAA representation of the time series.
        """
        n = len(X)
        if n == 0 or n_segments <= 0:
            return np.array([])
        if n % n_segments == 0:
            return X.reshape(n_segments, n // n_segments).mean(axis=1)
        else:
            segment_length = n / float(n_segments)
            paa = []
            for i in range(n_segments):
                start = int(np.ceil(i * segment_length))
                end = int(np.ceil((i + 1) * segment_length))
                if start >= n:
                    paa.append(0.0)
                else:
                    paa.append(X[start : min(end, n)].mean())
            return np.array(paa)

    sax_breakpoints = [-0.6745, 0, 0.6745]

    def extract_sax_candidates(self, data, subseq_length, sax_length=16):
        """
        Extract candidate shapelets using the SAX representation.

        This function slides a window over each time series, normalizes the subsequence,
        applies the PAA transformation, and then digitizes the result using predefined
        SAX breakpoints.

        Parameters
        ----------
        data : np.ndarray or list of arrays
            List of univariate time series.
        subseq_length : int
            Length of the subsequence (candidate shapelet).
        sax_length : int, default=16
            Number of segments for the PAA transformation used in the SAX
            representation.

        Returns
        -------
        candidates : list of tuples
            Each tuple has (series index, start index, SAX word as a tuple of integers)
        """
        candidates = []
        for series_idx, series in enumerate(data):
            series = np.squeeze(np.array(series))
            if len(series) < subseq_length:
                continue
            for start in range(0, len(series) - subseq_length + 1):
                subseq = series[start : start + subseq_length]
                norm_subseq = self.z_normalize(subseq)
                paa = self.paa_transform(norm_subseq, sax_length)
                # 0 for normlized value below -0.6745 (from brakpoints)
                # 1 for value between -0.6745 and 0.6745
                # 2 for values that are 0
                # 3 for values above 0.6745
                symbols = np.digitize(paa, self.sax_breakpoints)
                sax_word = tuple(int(sym) for sym in symbols)
                candidates.append((series_idx, start, sax_word))
        return candidates

    def count_collisions(self, candidates, sax_length=16, projections=10, mask_size=5):
        """
        Count collisions for candidate shapelets based on masked SAX words.

        A collision occurs when two candidates share the same masked SAX word. This
        method performs random masking of the SAX word positions and counts the number
        of unique time series (by series index) that collide.

        Parameters
        ----------
        candidates : list of tuples
            List of candidate shapelets, where each candidate is represented as a tuple
            (series index, start index, SAX word).
        sax_length : int, default=16
            Length of the SAX word.
        projections : int, default=10
            Number of random projections (maskings) to perform.
        mask_size : int, default=5
            Number of positions to mask in the SAX word.

        Returns
        -------
        collision_counts : list of list of int
            A list where each element is a list of collision counts for a candidate
            across projections.
        """
        sax_words = [cand[2] for cand in candidates]
        series_ids = [cand[0] for cand in candidates]
        num_candidates = len(candidates)
        collision_counts = [[0] * projections for _ in range(num_candidates)]

        for p in range(projections):
            mask_positions = self.random_state.choice(
                np.arange(sax_length), size=mask_size, replace=False
            )
            mask_positions = set(mask_positions)
            masked_map = defaultdict(set)  # automatic new key creation

            for idx, word in enumerate(sax_words):
                masked_key = tuple(
                    word[i] if i not in mask_positions else "*"
                    for i in range(sax_length)
                )
                masked_map[masked_key].add(series_ids[idx])

            for idx, word in enumerate(sax_words):
                masked_key = tuple(
                    word[i] if i not in mask_positions else "*"
                    for i in range(sax_length)
                )
                collision_counts[idx][p] = len(masked_map[masked_key])

        return collision_counts

    def compute_gap_score(self, shapelet, data, lb, ub):
        """
        Compute the gap score for a given shapelet.

        The gap score evaluates the separation between two groups of time series based
        on their distance to the shapelet. It computes the difference between the lower
        group's (DA) upper bound and the upper group's (DB) lower bound, defined by
        their means and standard deviations.

        Parameters
        ----------
        shapelet : np.ndarray
            Candidate shapelet (normalized subsequence).
        data : np.ndarray or list of arrays
            List of univariate time series.
        lb : int
            Lower bound for the number of time series in a group.
        ub : int
            Upper bound for the number of time series in a group.

        Returns
        -------
        gap : float
            The gap score computed for the shapelet.
        best_size : int or None
            The best split size (number of time series assigned to one cluster)
            corresponding to the gap score.
        """
        N = len(data)
        distances = []
        L = len(shapelet)
        for series in data:
            series = np.squeeze(np.array(series))
            if len(series) < L:
                dist = float("inf")
            else:
                min_dist = float("inf")
                for j in range(0, len(series) - L + 1):
                    sub = self.z_normalize(series[j : j + L])
                    d = np.linalg.norm(shapelet - sub)
                    if d < min_dist:
                        min_dist = d
                        if min_dist == 0:
                            break
                dist = min_dist
            distances.append(dist)

        distances = np.array(distances)
        sorted_dists = np.sort(distances)
        best_gap = -float("inf")
        best_size = None
        for sizeA in range(lb, min(ub, N - lb) + 1):
            if sizeA >= N:
                break
            DA = sorted_dists[:sizeA]
            DB = sorted_dists[sizeA:]
            if len(DA) < lb or len(DB) < lb:
                continue
            mA, mB = DA.mean(), DB.mean()
            sA, sB = DA.std(ddof=0), DB.std(ddof=0)
            gap = (mB - sB) - (mA + sA)
            if gap > best_gap:
                best_gap = gap
                best_size = sizeA

        return best_gap, best_size

    def get_best_u_shapelet(
        self, data, subseq_length, sax_length=16, projections=10, lb=2, ub=None
    ):
        """
        Find the best shapelet in the data that maximizes the gap score.

        This method extracts candidate shapelets, counts collisions using masked
        SAX words, and evaluates the gap score for each candidate. The candidate
        with the highest gap score is selected as the best shapelet.

        Parameters
        ----------
        data : np.ndarray or list of arrays
            List of univariate time series.
        subseq_length : int
            Length of the candidate shapelet.
        sax_length : int, default=16
            Length of the SAX word.
        projections : int, default=10
            Number of random projections (maskings) to perform.
        lb : int
            Lower bound for the number of time series in a group.
        ub : int, optional
            Upper bound for the number of time series in a group.
            If None, it is set based on the number of time series.

        Returns
        -------
        best_shapelet : np.ndarray or None
            The best shapelet (normalized subsequence) found, or None if no candidate
            is found.
        best_gap : float or None
            The gap score corresponding to the best shapelet, or None if no candidate
            is found.
        best_split : int or None
            The split (cluster size) corresponding to the best shapelet, or None if
            no candidate is found.
        best_loc : tuple or None
            Tuple (series index, start index) indicating the location of the best
            shapelet, or None if no candidate is found.
        """
        N = len(data)
        if ub is None:
            ub = max(0, N - 2)

        candidates = self.extract_sax_candidates(data, subseq_length, sax_length)
        if not candidates:
            return None, None, None, None
        total_candidates = len(candidates)

        mask_size = max(1, int(round(sax_length * 0.33)))
        collision_counts = self.count_collisions(
            candidates, sax_length, projections, mask_size
        )

        avg_collisions = np.mean(collision_counts, axis=1)
        valid_mask = (avg_collisions >= lb) & (avg_collisions <= (ub if ub > 0 else N))
        filtered_indices = np.nonzero(valid_mask)[0]
        if len(filtered_indices) == 0:
            filtered_indices = np.arange(total_candidates)

        collision_var = np.std(np.array(collision_counts)[filtered_indices], axis=1)
        sorted_order = filtered_indices[np.argsort(collision_var)]

        top_count = len(sorted_order)  # evaluate all
        top_indices = sorted_order[:top_count]

        best_gap = -float("inf")
        best_shapelet = None
        best_shapelet_loc = None
        best_split = None

        for idx in top_indices:
            series_idx, start, sax_word = candidates[idx]
            shapelet_seq = self.z_normalize(
                np.squeeze(np.array(data[series_idx]))[start : start + subseq_length]
            )
            gap, split_size = self.compute_gap_score(shapelet_seq, data, lb, ub)
            if gap > best_gap:
                best_gap = gap
                best_shapelet = shapelet_seq
                best_shapelet_loc = (series_idx, start)
                best_split = split_size

        return best_shapelet, best_gap, best_split, best_shapelet_loc

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """
        Return testing parameters for the UShapeletClusterer.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        params : dict
            Dictionary of parameters for testing.
        """
        return {
            "subseq_length": 3,
            "sax_length": 4,
            "projections": 5,
            "lb": 1,
            "ub": 4,
            "random_state": 18,
        }
