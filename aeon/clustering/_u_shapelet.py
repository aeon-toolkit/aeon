from collections import defaultdict
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state

from aeon.clustering._k_means import TimeSeriesKMeans
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
    n_clusters : int, optional, default=2
        Number of clusters to form.
        If 2 (or None), a single best shapelet is used to split the data into
        two clusters.
        If > 2, a set of shapelets is extracted to build a distance map
        for subsequent K-means clustering.
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
        n_clusters: Optional[int] = 2,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.subseq_length = subseq_length
        self.sax_length = sax_length
        self.projections = projections
        self.lb = lb
        self.ub = ub
        self.n_clusters = n_clusters
        # Store the random_state parameter as provided.
        self.random_state = random_state

        self.best_shapelet_ = None
        self.best_gap_ = None
        self.best_split_ = None
        self.best_loc_ = None

        # for multi shapelets
        self._multi_shapelets = None
        self._kmeans_model = None
        self._kmeans_labels = None

        super().__init__()

    def _fit(self, X: np.ndarray, y=None) -> "UShapeletClusterer":
        """
        Fit the UShapeletClusterer to the provided time series data.

        If n_clusters <= 2 or None, performs single-shapelet discovery to produce
        a two-cluster split. Otherwise, a multiple-shapelet approach is used to
        generate a distance map, which is then clustered using K-means.

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
        X = np.array(X, dtype=object)
        N = len(X)

        # n_clusters=2 => single shapelet => no kmeans
        if self.n_clusters is None or self.n_clusters == 2:
            (best_shapelet, best_gap, best_split, best_loc) = self.get_best_u_shapelet(
                X,
                self.subseq_length,
                sax_length=self.sax_length,
                projections=self.projections,
                lb=self.lb,
                ub=self.ub,
            )
            self.best_shapelet_ = best_shapelet
            self.best_gap_ = best_gap
            self.best_split_ = best_split
            self.best_loc_ = best_loc

            # no multi-shapelets, no kmeans
            self._multi_shapelets = []
            self._kmeans_model = None
            self._kmeans_labels = None

        else:
            # n_clusters>2 => multi-shapelet distance map => time series k-means
            self._multi_shapelets = self._extract_multiple_shapelets(X)

            if len(self._multi_shapelets) == 0:
                # fallback: no shapelets => all label 0
                self._kmeans_labels = np.zeros(N, dtype=int)
                return self

            # compute distance map => NxM
            dist_map = self._compute_distance_map(X, self._multi_shapelets)

            self._kmeans_model = TimeSeriesKMeans(
                n_clusters=self.n_clusters,
                distance="euclidean",
                max_iter=10,
                random_state=self.random_state,
            )
            self._kmeans_model.fit(dist_map)
            self._kmeans_labels = self._kmeans_model.labels_

        return self

    def _predict(self, X, y=None):
        """
        Predict cluster labels for each time series in X.

        - If using 2 clusters (single shapelet), each series is assigned 0 or 1 based on
          whether its minimal distance to the best shapelet is below or above the
          learned threshold.
        - If using > 2 clusters, a distance map is built from the multi-shapelets
          and passed to the fitted k-means model.

        Parameters
        ----------
        X : np.ndarray or list of arrays
            Univariate time series to predict.
        y : None
            Ignored.

        Returns
        -------
        labels : np.ndarray
            Array of cluster labels for each time series.
        """
        X = np.array(X, dtype=object)

        if self.n_clusters is None or self.n_clusters == 2:
            if self.best_shapelet_ is None or self.best_split_ is None:
                return np.zeros(len(X), dtype=int)
            dists = self._compute_min_dist_per_series(X, self.best_shapelet_)
            threshold = np.sort(dists)[self.best_split_ - 1]
            return np.array([0 if d <= threshold else 1 for d in dists])
        else:
            if self._multi_shapelets is None or self._kmeans_model is None:
                return np.zeros(len(X), dtype=int)
            dist_map = self._compute_distance_map(X, self._multi_shapelets)
            return self._kmeans_model.predict(dist_map)

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

        This function slides a window of length ``subseq_length`` over each time series,
        normalizes the subsequence, applies PAA of length ``sax_length``, and digitizes
        the result using predefined SAX breakpoints. Each resulting subsequence is
        treated as a candidate shapelet.

        Parameters
        ----------
        data : np.ndarray or list of arrays
            List of univariate time series.
        subseq_length : int
            Length of each candidate subsequence (shapelet).
        sax_length : int, default=16
            Number of segments for the PAA transform used in the SAX representation.

        Returns
        -------
        candidates : list of tuples
            Each tuple is (series_idx, start_index, sax_word),
            where sax_word is a tuple of integers representing the digitized PAA.
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
                # 0 for normalized value < -0.6745,
                # 1 for value between -0.6745 and 0,
                # 2 for value between 0 and 0.6745,
                # 3 for value >= 0.6745
                symbols = np.digitize(paa, self.sax_breakpoints)
                sax_word = tuple(int(sym) for sym in symbols)
                candidates.append((series_idx, start, sax_word))
        return candidates

    def count_collisions(self, candidates, sax_length=16, projections=10, mask_size=5):
        """
        Count collisions for candidate shapelets based on masked SAX words.

        A collision occurs when two subsequence candidates produce the
        same masked SAX word for a given random projection (mask).
        This method applies multiple random masks, each time counting the
        distinct time series that collide on the masked positions.

        Parameters
        ----------
        candidates : list of tuples
            Each candidate is (series_idx, start_index, sax_word).
        sax_length : int, default=16
            Length of the original SAX word.
        projections : int, default=10
            Number of random projections (maskings) to perform.
        mask_size : int, default=5
            Number of positions in the SAX word to mask.

        Returns
        -------
        collision_counts : list of lists
            A list (parallel to candidates), where each element is
            a list of length ``projections`` storing the collision count
            of that candidate for each random mask.
        """
        sax_words = [cand[2] for cand in candidates]
        series_ids = [cand[0] for cand in candidates]
        num_candidates = len(candidates)
        collision_counts = [[0] * projections for _ in range(num_candidates)]

        # input random_state to a proper random state object.
        rs = check_random_state(self.random_state)

        for p in range(projections):
            mask_positions = rs.choice(
                np.arange(sax_length), size=mask_size, replace=False
            )
            mask_positions = set(mask_positions)
            masked_map = defaultdict(set)

            # Build a dictionary from masked_key -> set of series indices
            for idx, word in enumerate(sax_words):
                masked_key = tuple(
                    word[i] if i not in mask_positions else "*"
                    for i in range(sax_length)
                )
                masked_map[masked_key].add(series_ids[idx])

            # For each candidate, record how many distinct series share its masked_key
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

        The gap score measures how well a shapelet can split the dataset
        into two groups by distance. For each time series, we compute its
        minimal distance to the shapelet. Sorting these distances, we try
        every possible split size in [lb, ub], computing:

            gap = (mean(DB) - std(DB)) - (mean(DA) + std(DA)),

        where DA is the left side (closest time series) and DB is the right side.

        Parameters
        ----------
        shapelet : np.ndarray
            Candidate shapelet (z-normalized subsequence).
        data : np.ndarray or list of arrays
            Collection of univariate time series.
        lb : int
            Lower bound on cluster size to consider.
        ub : int
            Upper bound on cluster size to consider.

        Returns
        -------
        gap : float
            The maximum gap score found among all split points.
        best_size : int
            The size of left cluster (DA) for the best gap, or None if no valid split.
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

    def _extract_multiple_shapelets(
        self, X: np.ndarray, fraction: float = 0.01, max_shapelets: int = 10
    ):
        """
        Discover multiple shapelets for building a distance map (for k-means).

        This method collects a large set of candidate shapelets via SAX, filters them
        based on collision counts (removing stop-words or outliers), then sorts them
        by the variance of collision counts. It then evaluates a small fraction
        (given by 'fraction') of these in terms of their gap scores to find
        discriminative candidates. Finally, it keeps up to 'max_shapelets' of the
        best ones.

        Parameters
        ----------
        X : np.ndarray or list of arrays
            The dataset of univariate time series.
        fraction : float, default=0.01
            The fraction of filtered candidates to evaluate by gap score.
            For example, 0.01 means we only check 1% of the best collision-variance
            candidates in detail.
        max_shapelets : int, default=10
            The maximum number of shapelets to retain after gap-score evaluation.

        Returns
        -------
        list of np.ndarray
            The final set of shapelet subsequences chosen for building the
            distance map. Each shapelet is a z-normalized 1D numpy array.
        """
        N = len(X)
        if self.ub is None:
            self.ub = max(0, N - 2)

        candidates = self.extract_sax_candidates(X, self.subseq_length, self.sax_length)
        if not candidates:
            return []

        total_candidates = len(candidates)

        mask_size = max(1, int(round(self.sax_length * 0.33)))
        collisions = self.count_collisions(
            candidates, self.sax_length, self.projections, mask_size
        )

        avg_colls = np.mean(collisions, axis=1)
        valid_mask = (avg_colls >= self.lb) & (
            avg_colls <= (self.ub if self.ub > 0 else N)
        )
        filtered_indices = np.nonzero(valid_mask)[0]
        if len(filtered_indices) == 0:
            filtered_indices = np.arange(total_candidates)

        colls_var = np.std(np.array(collisions)[filtered_indices], axis=1)
        sorted_order = filtered_indices[np.argsort(colls_var)]

        # evaluate only top fraction
        top_count = max(1, int(total_candidates * fraction))
        top_idx = sorted_order[:top_count]

        shapelets = []
        for idx in top_idx:
            s_idx, start, sax_word = candidates[idx]
            shape = self.z_normalize(
                np.squeeze(X[s_idx])[start : start + self.subseq_length]
            )
            gap, split_size = self.compute_gap_score(shape, X, self.lb, self.ub)
            shapelets.append((gap, shape))

        shapelets.sort(key=lambda x: x[0], reverse=True)
        shapelets = shapelets[:max_shapelets]

        final = [shp for (gp, shp) in shapelets if gp != -float("inf")]
        return final

    def _compute_distance_map(self, X, shapelets):
        """
        Construct an N x M distance matrix from multiple shapelets.

        Each row corresponds to a time series, and each column is the minimal
        Euclidean distance to one shapelet.

        Parameters
        ----------
        X : np.ndarray or list of arrays
            The dataset of univariate time series.
        shapelets : list of np.ndarray
            A set of shapelet subsequences.

        Returns
        -------
        dist_map : np.ndarray
            Array of shape (N, M), where N=len(X) and M=len(shapelets).
            dist_map[i, j] is the min distance between series i and shapelet j.
        """
        N = len(X)
        M = len(shapelets)
        dist_map = np.zeros((N, M), dtype=float)

        for m, shape in enumerate(shapelets):
            dists = self._compute_min_dist_per_series(X, shape)
            dist_map[:, m] = dists
        return dist_map

    def _compute_min_dist_per_series(self, X, shapelet):
        """
        Compute the minimum distance of each series to a given shapelet.

        For each time series in X, slides a window matching the shapelet length,
        z-normalizes each subsequence, and records the minimal Euclidean distance
        to the shapelet.

        Parameters
        ----------
        X : np.ndarray or list of arrays
            The dataset of univariate time series.
        shapelet : np.ndarray
            A single shapelet subsequence (z-normalized) to compare against.

        Returns
        -------
        np.ndarray
            A 1D array of shape (N,) containing minimal distances to 'shapelet'
            for each time series in X.
        """
        L = len(shapelet)
        dists = np.zeros(len(X), dtype=float)

        for i, series in enumerate(X):
            series = np.squeeze(series)
            if len(series) < L:
                dists[i] = float("inf")
                continue
            min_dist = float("inf")
            for j in range(0, len(series) - L + 1):
                sub = self.z_normalize(series[j : j + L])
                dd = np.linalg.norm(shapelet - sub)
                if dd < min_dist:
                    min_dist = dd
            dists[i] = min_dist

        return dists

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
