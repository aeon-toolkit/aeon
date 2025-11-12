from __future__ import annotations

import collections
import math
import random
import time
from multiprocessing import get_context
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from aeon.clustering import BaseClusterer
from aeon.utils.validation import check_n_jobs

ArrayLike = Union[np.ndarray, list[np.ndarray]]


class KGraphClusterer(BaseClusterer):
    """kGraph time-series clusterer wrapped in the BaseClusterer interface.

    Notes
    -----
    - This adapter delegates to the original `kGraph` engine for training.
    - The original implementation clusters on the *training set* and does not
      provide a robust out-of-sample prediction for unseen series. Therefore,
      `predict(X)` raises `NotImplementedError`. Use `fit_predict(X)` (or `fit`
      followed by reading `labels_`) as with many graph/consensus methods.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    n_lengths : int, default=10
        Number of segment lengths to evaluate.
    n_jobs : int, default=1
        Parallel jobs for graph computation per length.
    rate : int, default=30
        Number of radii in radial scan (controls node density).
    min_length : int, default=10
        Minimum pattern length.
    rate_max_length : float, default=0.33
        Max pattern length is `rate_max_length * min_series_length`.
    seed : int, default=0
        Random seed.
    sample : int, default=10
        Subsampling factor for PCA training in the projection step.
    variable_length : bool, default=False
        If True, X may be a list of arrays with different lengths.
    precompute_explaination : bool, default=False
        Whether to precompute optimal length for interpretation.
    verbose : bool, default=True
        Verbosity of the underlying engine.
    """

    _tags = {
        "fit_is_empty": False,
        # Optional extra tags (uncomment if your framework uses them):
        # "capability:multivariate": "partial",   # kGraph expects univariate; we downselect ch=0
        # "capability:unequal_length": True,
        # "X_inner_mtype": ["numpy3D", "numpy2D", "np-list"],  # aeon style
    }

    def __init__(
        self,
        n_clusters: int = 2,
        n_lengths: int = 30,  # M = 30 (default number of lengths)
        n_jobs: int = 1,
        rate: int = 30,  # number of radii (used in all experiments)
        min_length: int = 5,  # minimum subsequence length (paper: at least 5 points)
        rate_max_length: float = 0.4,  # rml = 0.4 (default max-length rate)
        seed: int = 0,
        sample: int = 10,  # smpl = 10 (default sample rate)
        variable_length: bool = False,
        precompute_explaination: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_lengths = n_lengths
        self.n_jobs = n_jobs
        self.rate = rate
        self.min_length = min_length
        self.rate_max_length = rate_max_length
        self.seed = seed
        self.sample = sample
        self.variable_length = variable_length
        self.precompute_explaination = precompute_explaination
        self.verbose = verbose

        self._engine: kGraph | None = None
        self.labels_: np.ndarray | None = None
        self.all_lengths: list[int] | None = None
        self.graphs = None
        self.length_relevance = None
        self.graph_relevance = None
        self.relevance = None
        self.optimal_length: int | None = None

    # ------------------------- BaseClusterer hooks -------------------------

    def _fit(self, X: ArrayLike, y=None):
        """Train the kGraph engine and store labels on the training set."""
        n_jobs = check_n_jobs(self.n_jobs)
        X_list, variable_length = _to_univariate_list(X)
        # honour explicit user choice if set; otherwise infer
        use_variable_len = self.variable_length or variable_length

        self._engine = kGraph(
            n_clusters=self.n_clusters,
            n_lengths=self.n_lengths,
            n_jobs=n_jobs,
            rate=self.rate,
            min_length=self.min_length,
            rate_max_length=self.rate_max_length,
            seed=self.seed,
            sample=self.sample,
            variable_length=use_variable_len,
            precompute_explaination=self.precompute_explaination,
            verbose=self.verbose,
        )
        self._engine.fit(X_list)

        # expose key learned attributes
        self.labels_ = np.asarray(self._engine.labels_, dtype=int)
        self.graphs = self._engine.graphs
        self.all_lengths = list(self._engine.graphs.keys())
        self.length_relevance = getattr(self._engine, "length_relevance", None)
        self.graph_relevance = getattr(self._engine, "graph_relevance", None)
        self.relevance = getattr(self._engine, "relevance", None)
        self.optimal_length = getattr(self._engine, "optimal_length", None)

        return self

    def _predict(self, X: ArrayLike) -> np.ndarray:
        """Out-of-sample predict is not supported for kGraph."""
        raise NotImplementedError(
            "kGraph provides clustering only on the fitted dataset. "
            "Use `fit_predict(X)` (or read `labels_` after `fit`)."
        )

    # ---------------------- Convenience passthroughs -----------------------

    def interprete(self, length: int | None = None, nb_patterns: int = 1):
        """Delegate to engine's interpretation (most representative nodes per cluster)."""
        self._check_is_fitted()
        return self._engine.interprete(length=length, nb_patterns=nb_patterns)

    def compute_graphoids(
        self,
        length: int | None = None,
        mode: str = "Exclusive",
        majority_level: float = 0.8,
    ):
        """Delegate to engine's graphoid computation."""
        self._check_is_fitted()
        return self._engine.compute_graphoids(
            length=length, mode=mode, majority_level=majority_level
        )

    def get_node_ts(self, X: ArrayLike, node: str, length: int | None = None):
        """Delegate to engine's representative time-series extraction for a node."""
        self._check_is_fitted()
        X_list, _ = _to_univariate_list(X)
        return self._engine.get_node_ts(X_list, node, length=length)


# ----------------------------- helpers ---------------------------------


def _to_univariate_list(X: ArrayLike) -> tuple[list[np.ndarray], bool]:
    """Convert equal/unequal-length inputs into a list of 1D numpy arrays.

    Returns
    -------
    list_of_1d : list[np.ndarray]
        Each element is shape (L_i,), dtype=float64.
    variable_length : bool
        True if input sequences have unequal lengths.
    """
    # list input (possibly unequal length)
    if isinstance(X, list):
        seqs: list[np.ndarray] = []
        lengths = []
        for xi in X:
            arr = np.asarray(xi)
            if arr.ndim == 2:
                # take first channel for multivariate
                arr = arr[0]
            elif arr.ndim != 1:
                raise ValueError(
                    "kGraph expects univariate series; got element with shape "
                    f"{arr.shape}. Provide univariate or handle channel selection upstream."
                )
            seqs.append(np.ascontiguousarray(arr, dtype=float))
            lengths.append(len(arr))
        variable = len(set(lengths)) > 1
        return seqs, variable

    # numpy array input
    arr = np.asarray(X)
    if arr.ndim == 3:
        # (n_cases, n_channels, n_timepoints) -> select channel 0
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(
            "kGraph expects 2D (n_cases, n_timepoints), 3D (n_cases, n_channels, n_timepoints), "
            "or list-of-arrays. Got shape: {}".format(arr.shape)
        )
    seqs = [np.ascontiguousarray(arr[i], dtype=float) for i in range(arr.shape[0])]
    return seqs, False


LIST_COLOR = [
    "#ff8c00",
    "#0000ff",
    "#808000",
    "#ff0000",
    "#00ffff",
    "#f0ffff",
    "#f5f5dc",
    "#000000",
    "#a52a2a",
    "#00ffff",
    "#00008b",
    "#008b8b",
    "#a9a9a9",
    "#006400",
    "#bdb76b",
    "#8b008b",
    "#556b2f",
    "#9932cc",
    "#8b0000",
    "#e9967a",
    "#9400d3",
    "#ff00ff",
    "#ffd700",
    "#008000",
    "#4b0082",
    "#f0e68c",
    "#add8e6",
    "#e0ffff",
    "#90ee90",
    "#d3d3d3",
    "#ffb6c1",
    "#ffffe0",
    "#00ff00",
    "#ff00ff",
    "#800000",
    "#000080",
    "#808000",
    "#ffa500",
    "#ffc0cb",
    "#800080",
    "#800080",
    "#c0c0c0",
    "#ffffff",
    "#ffff00",
]


class kGraph:
    """
    kGraph method for time series clustering.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters (default is 2).

    n_lengths : int, optional
        Number of lengths (default is 10).
        The lengths are selected between min_length and the time series length
        multiplied by rate_max_length.

    n_jobs : int, optional
        Number of jobs tun run in parallel the graph computation for each length
        (default is 1).

    rate : int, optional
        Number of radius for the radial scan of the node creation step (default is 30).
        This parameter controls the number of nodes per graph. A low rate will
        limit the number of nodes.

    min_length : int, optional
        Minimum length (default is 10).

    rate_max_length : float, optional
        Rate defining the maximum length (default is 0.33).
        This rate is multiplied by the time series lengths.

    seed : int, optional
        Seed for reproducibility (default is 0).

    sample : int, optional
        Sample parameter to train the PCA of the projection step (default is 10).

    variable_length : bool, optional
        If True, X is expected to be a list of arrays of differnt lengths (default is False).
        If False, X is expected to be an array.

    precompute_explaination: bool, optional
        If True, precompute optimal length used for interpretation of the clustering (default False).
        If False, optimal length will be computed at each explanation/interpretation.
        Recommended to set to True if an interpretation/explanation is desired.
        Recommended to set to False if only the clustering results is needed.

    verbose : bool, optional
        If True, print verbose information during execution (default is True).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each time series

    all_lengths : ndarray of shape (n_lengths,)
        The selected length. The lengths are contained between min_length and
        the time series length multiplied by rate_max_length.

    length_relevance : ndarray of shape (2,n_lengths)
        Relevance of each length according to the consensus clustering.
        In practice, the relevance is computed by measuring the Adjusted Rand Index
        between the consensus clustering and the kmeans clustering from the graph
        corresponding to the length.

    graph_relevance : ndarray of shape (2,n_lengths)
        Relevance of each length according to the corresponding graph interpretability.

    relevance : ndarray of shape (2,n_lengths)
        Relevance of each length according to the corresponding graph interpretability
        and the consensus clustering. Relevance is the element-wise product between
        the length_relevance and the graph_relevance.

    optimal_length : int
        The optimal length based on the relevance

    graphs : dict with n_lengths as keys
        information stored for each graph. Each key/length is associated to a dict
        with the following keys:

        - graph: dict containing information on the graph, with the following keys:
            - list_edge : List of successive edges.

            - dict_edge : dict with the edges as keys and the number of times the
                edges have been crossed as values.

            - dict_node : dict with the nodes as keys and the number of times the
                nodes have been crossed as values.

            - list_edge_pos : list of int with a length equal to n_samples.
                list_edge_pos[i] and list_edge_pos[i+1] corresponds to the position
                in list_edge when the time series i starts and ends.

            - edge_in_time : list of lists with a length equal to n_samples
                Each list corresponds to one time series T and has the same length
                as T. The position edge_in_time[i][j] corresponds to the position
                in list_edge at the timestamp j in the time series i.

        - prediction: ndarray of shape (n_samples,)
            Labels of each time series according to the graph associated to the
            length (i.e., the key).

        - feature : pandas DataFrame (n_samples, n_nodes+n_edges)
            DataFrame containing the feature from the graph (used for the first
            clustering step) for each time series.
    """

    def __init__(
        self,
        n_clusters=2,
        n_lengths=10,
        n_jobs=1,
        rate=30,
        min_length=10,
        rate_max_length=0.33,
        seed=0,
        sample=10,
        variable_length=False,
        precompute_explaination=False,
        verbose=True,
    ):
        """
        Initialize kGraph method
        """
        self.n_clusters = n_clusters
        self.n_lengths = n_lengths
        self.n_jobs = n_jobs

        self.rate = rate
        self.min_length = min_length
        self.rate_max_length = rate_max_length
        self.seed = seed
        self.verbose = verbose
        self.sample = sample
        self.variable_length = variable_length
        self.compute_revelance = precompute_explaination

        self.optimal_length = None

    # Public method

    def fit(self, X, y=None):
        """
        Compute kGraph on X

        Parameters
        ----------
        X : array of shape (n_samples, n_timestamps)
            Training instances to cluster.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        length = min([len(x) for x in X])
        random_length_set = list(
            {
                int(val)
                for val in set(
                    np.linspace(
                        self.min_length,
                        max(self.min_length, int(length * self.rate_max_length)),
                        self.n_lengths,
                    )
                )
            }
        )

        self.__verboseprint(
            f"Running kGraph for the following length: {random_length_set}"
        )

        tim_start = time.time()

        if self.sample > (len(X) * (len(X[0]) - max(random_length_set))) // 3:
            self.sample = (len(X) * (len(X[0]) - max(random_length_set))) // 3
            self.__verboseprint(
                "[WARNING]: Sample too large. Setting to the maximum acceptable value: {}".format(
                    self.sample
                )
            )

        parameters = [[X, pattern_length] for pattern_length in random_length_set]
        with get_context("spawn").Pool(processes=self.n_jobs) as pool:
            all_pred_raw = pool.map(self.run_graphs_parallel, parameters)

        all_pred, all_graph, all_df, all_pattern = [], [], [], []

        for pred in all_pred_raw:
            all_pred.append(np.array(pred[0]))
            all_graph.append(pred[1])
            all_df.append(pred[2])
            all_pattern.append(pred[3])

        self.__verboseprint(f"Graphs computation done! ({time.time() - tim_start} s)")

        tim_start = time.time()
        sim_matrix = self.__build_consensus_matrix(all_pred)

        self.__verboseprint(f"Consensus done! ({time.time() - tim_start} s)")

        tim_start = time.time()

        clustering_ens = SpectralClustering(
            n_clusters=self.n_clusters,
            assign_labels="discretize",
            affinity="precomputed",
            random_state=self.seed,
            n_jobs=self.n_jobs,
        ).fit(sim_matrix)

        self.__verboseprint(f"Ensemble clustering done! ({time.time() - tim_start} s)")

        self.graphs = {
            all_pattern[i]: {
                "graph": all_graph[i],
                "prediction": all_pred[i],
                "feature": all_df[i],
            }
            for i in range(len(random_length_set))
        }
        self.graphs = collections.OrderedDict(sorted(self.graphs.items()))
        self.all_lengths = list(self.graphs.keys())
        self.labels_ = clustering_ens.labels_

        if self.compute_revelance == True:
            self.__get_length_relevance()

        return self

    def compute_graphoids(self, length=None, mode="Exclusive", majority_level=0.8):
        """
        Extract graphoid for a given length. More precisely, the graphoids associated to
        each node and edge the number of time series of a given cluster that crossed it.

        Parameters
        ----------
        length : int or None, optional
            Length parameter for graphoid computation (default is None).
            If None, length is set to self.optimal_length.

        mode : {'Raw','Exclusive', 'Proportion'}, optional
            Mode for computing graphoids. If 'Exclusive', compute exclusive graphoids
            (i.e., set to zero the nodes and edges that are not above majority_level).
            If 'Proportion', compute graphoids and normalize the value such that, for one node,
            the sum for all graphoids is equal to one. (default is 'Exclusive').

        majority_level : float, optional
            Majority level threshold for graphoid computation (default is 0.8).
            Not used for Proportion mode

        Returns
        -------
        result : ndarray of shape (n_clusters,n_nodes + n_edges)
            Graphoids for each cluster.

        """
        if length is None:
            if self.optimal_length is None:
                self.__get_length_relevance()
            length = self.optimal_length

        all_graphoid, names_features = self.__compute_all_graphoid(length)
        if mode == "Raw":
            pass
        elif mode == "Exclusive":
            all_graphoid = self.__compute_all_exclusive_graphoid(
                all_graphoid, majority_level=majority_level
            )
        elif mode == "Proportion":
            all_graphoid = self.__compute_all_prop_graphoid(
                all_graphoid, length, names_features, majority_level=0.8
            )

        return np.array(all_graphoid), names_features

    def get_node_ts(self, X, node, length=None):
        """
        For a given node (for a given dataset X), compute the representative time series of the node.
        X has to be the same as the one used in the fit method.

        Parameters
        ----------
        X : array of shape (n_samples, n_timestamps)
            Instances used to train kGraph in the fit method.

        node : str
            Must be be an existing node in self.graphs[length]['graph']['dict_node']

        length : int or None, optional
            Length parameter for graphoid computation (default is None).
            If None, length is set to self.optimal_length.

        Returns
        -------
        result : three 1D arrays of shape (length)
            return the average representative, the upper-bound (mean + std)
            and the lower bound (mean - std). Note that the returned time series
            is normalized.

        """
        if length is None:
            if self.optimal_length is None:
                self.__get_length_relevance()
            length = self.optimal_length

        result = []

        global_pos = []
        current_pos = 0

        edge_in_time = self.graphs[length]["graph"]["edge_in_time"]
        for i, edge in enumerate(self.graphs[length]["graph"]["list_edge"]):

            if node == edge[0]:
                relative_pos = (
                    i - self.graphs[length]["graph"]["list_edge_pos"][current_pos]
                )
                pos_in_time = min(
                    range(len(edge_in_time[current_pos])),
                    key=lambda j: abs(edge_in_time[current_pos][j] - relative_pos),
                )
                if self.variable_length:
                    ts = X[int(current_pos)][
                        int(pos_in_time) : int(pos_in_time + length)
                    ]
                else:
                    ts = X[
                        int(current_pos), int(pos_in_time) : int(pos_in_time + length)
                    ]
                ts = ts - np.mean(ts)
                result.append(ts)

            if i >= self.graphs[length]["graph"]["list_edge_pos"][current_pos + 1]:
                current_pos += 1

        mean = np.mean(result, axis=0)
        dev = np.std(result, axis=0)

        return mean, mean - dev, mean + dev

    def interprete(self, length=None, nb_patterns=1):
        """
        Return the nb_patterns most representative nodes (i.e., the nodes
        that have been crossed by most of (and only) time series of a
        given cluster).

        Parameters
        ----------
        length : int or None, optional
            Length parameter for graphoid computation (default is None).
            If None, length is set to self.optimal_length.

        nb_patterns: int, optional (default is 1)
            Number of nodes to return.

        Returns
        -------
        result : dict with the cluster labels as keys.
            Each cluster C is associated with a list of nb_patterns nodes
            (sorted by their representativeness and effectiveness).
            Each node is a list with the following elements:

            - Node name: str
                name of the node in the graph associated with length.

            - Importance score: float (between 0 and 1)
                Exclusivity and Representativity score multiplied.

            - Exclusivity score: float (between 0 and 1)
                Percentage of time series crossing the node that belongs to cluster C.

            - Representativity score: float (between 0 and 1)
                Percentage of time series of cluster C crossing the node.

        """
        if length is None:
            if self.optimal_length is None:
                self.__get_length_relevance()
            length = self.optimal_length

        all_graphoid, names_features = self.compute_graphoids(
            length=length, mode="Proportion"
        )
        names_features = np.array(names_features)

        nodes_name = []
        for i in range(len(names_features)):
            if "[" not in names_features[i]:
                nodes_name.append(i)

        cluster_interpretation = {}
        for cluster_x in set(self.labels_):
            norm_graph = all_graphoid[cluster_x]

            set_nodes = []
            nodes_exclusivity = []

            max_exp_node = 0.8

            for val, name in zip(norm_graph[nodes_name], names_features[nodes_name]):
                # if val > max_exp_node:
                set_nodes.append(name)
                nodes_exclusivity.append(val)

            nodes_representativity = self.__compute_representativity_node(
                length, set_nodes, cluster_x
            )

            cluster_interpretation[cluster_x] = sorted(
                [
                    [node, exc * rep, exc, rep]
                    for node, exc, rep in zip(
                        set_nodes, nodes_exclusivity, nodes_representativity
                    )
                ],
                key=lambda tup: tup[1],
                reverse=True,
            )[:nb_patterns]
        return cluster_interpretation

    def explain(self, i_x, length=None):
        """
        Compute the local explanation curve for time series i_x in X.
        X is the same as the one used in the fit method.

        Parameters
        ----------
        i_x : int.
            Index of the time series to explain in X.

        length : int or None, optional
            Length parameter for graphoid computation (default is None).
            If None, length is set to self.optimal_length.

        Returns
        -------
        result : 1D array of shape (X[i_x].shape[1],)
            return a time series (same length as the time series i_x in X).
            A high value in the latter indicates that the corresponding position in T
            is a representative element of T of belonging to cluster self.labels_[i_x].

        """
        if length is None:
            if self.optimal_length is None:
                self.__get_length_relevance()
            length = self.optimal_length

        edge_in_time = self.graphs[length]["graph"]["edge_in_time"][i_x]
        edges_start = self.graphs[length]["graph"]["list_edge_pos"][i_x]
        edges_end = self.graphs[length]["graph"]["list_edge_pos"][i_x + 1]
        edges = self.graphs[length]["graph"]["list_edge"][edges_start:edges_end]

        random_prob = 1.0 / float(self.n_clusters)
        cluster_x = self.labels_[i_x]
        all_graphoid, names_features = self.compute_graphoids(
            length=length, mode="Proportion"
        )

        all_nodes = [[]]
        all_edges = [[]]
        for i in range(1, len(edge_in_time)):
            edges_seq = edges[edge_in_time[i - 1] : edge_in_time[i]]
            if len(edges_seq) > 0:
                nodes_seq = [e[0] for e in edges_seq] + [edges_seq[-1][1]]
            else:
                if len(all_nodes[-1]) > 0:
                    nodes_seq = [all_nodes[-1][-1]]
                else:
                    nodes_seq = []
            all_nodes.append(nodes_seq)
            all_edges.append(edges_seq)

        norm_graph = all_graphoid[cluster_x]

        all_node_relevance, all_edge_relevance = [], []

        for i, (nodes_seq, edges_seq) in enumerate(zip(all_nodes, all_edges)):
            if len(nodes_seq) > 0:
                node_relevance = np.mean(
                    [norm_graph[names_features.index(node)] for node in nodes_seq]
                )
            else:
                node_relevance = random_prob
            if len(edges_seq) > 0:
                edge_relevance = np.mean(
                    [norm_graph[names_features.index(str(edge))] for edge in edges_seq]
                )
            else:
                edge_relevance = random_prob

            all_node_relevance.append(node_relevance)
            all_edge_relevance.append(edge_relevance)

        return np.max([all_node_relevance, all_edge_relevance], axis=0)

    # Private method for kGraph computation

    def __compute_representativity_node(self, length, list_elem, label):

        res = {elem: 0 for elem in list_elem}

        tot_ts = len(self.graphs[length]["graph"]["list_edge_pos"]) - 1

        cluster_prop = list(self.labels_).count(label)

        for i in range(tot_ts):
            if self.labels_[i] == label:
                edges_start = self.graphs[length]["graph"]["list_edge_pos"][i]
                edges_end = self.graphs[length]["graph"]["list_edge_pos"][i + 1]
                edges = self.graphs[length]["graph"]["list_edge"][edges_start:edges_end]

                if len(edges) > 0:
                    nodes = set([e[0] for e in edges] + [edges[-1][1]])
                else:
                    nodes = []

                for elem in list_elem:
                    if elem in nodes:
                        res[elem] += 1 / cluster_prop

        return list(res.values())

    def __get_length_relevance(self):

        self.length_relevance = []
        self.graph_relevance = []
        for length in self.graphs.keys():
            all_graphoid, names_features = self.compute_graphoids(
                length=length, mode="Proportion"
            )
            self.length_relevance.append(
                adjusted_rand_score(self.graphs[length]["prediction"], self.labels_)
            )
            self.graph_relevance.append(
                np.mean(
                    np.max(all_graphoid[:, self.__get_node(names_features)], axis=1)
                )
            )

        self.relevance = np.array(self.length_relevance) * np.array(
            self.graph_relevance
        )

        self.length_relevance = np.array(
            [[l, val] for l, val in zip(self.graphs.keys(), self.length_relevance)]
        )
        self.graph_relevance = np.array(
            [[l, val] for l, val in zip(self.graphs.keys(), self.graph_relevance)]
        )
        self.relevance = np.array(
            [[l, val] for l, val in zip(self.graphs.keys(), self.relevance)]
        )

        self.optimal_length = self.relevance[np.argmax(self.relevance, axis=0)[1]][0]

    def __get_node(self, feature_names):
        res = []
        for i, name in enumerate(feature_names):
            if "[" not in name:
                res.append(i)
        return res

    def __compute_all_exclusive_graphoid(self, all_graphoid, majority_level=0.8):
        all_graphoid_exclusive = []
        for i, graphoid in enumerate(all_graphoid):
            tmp = np.sum(all_graphoid, axis=0)
            result = []
            for c, val in zip(graphoid, tmp):
                if c / (val + 0.0001) > majority_level:
                    result.append(2 * c - val)
                else:
                    result.append(0)
            all_graphoid_exclusive.append(result)
        return all_graphoid_exclusive

    def __compute_all_prop_graphoid(
        self, all_graphoid, length, names_features, majority_level=0.8
    ):
        all_graphoid_prop = []
        for i, graphoid in enumerate(all_graphoid):
            tmp = np.sum(all_graphoid, axis=0)
            result = []
            for c, val, name in zip(graphoid, tmp, names_features):
                result.append(c / (val + 0.0001))
            all_graphoid_prop.append(result)
        return all_graphoid_prop

    def __compute_all_graphoid(self, length):
        all_graphoid = []
        for cluster in set(self.labels_):
            graphoid, names_features = self.__compute_graphroid(length, cluster)
            all_graphoid.append(graphoid)
        return all_graphoid, names_features

    def __compute_graphroid(self, length, cluster):
        features = self.graphs[length]["feature"]
        data = []
        for i in range(len(self.labels_)):
            if cluster == self.labels_[i]:
                data.append(list(features.values[i]))
        data = np.array(data)
        return np.mean(data, 0), list(features.columns)

    def run_graphs_parallel(self, args):

        return self.__run_clustering_length(*args)

    def __create_dataset(self, G, X):
        df_node = pd.DataFrame(
            index=list(range(len(X))), columns=list(G["dict_node"].keys())
        )
        df_edge = pd.DataFrame(
            index=list(range(len(X))),
            columns=[str(edge) for edge in list(G["dict_edge"].keys())],
        )
        df_conf = pd.DataFrame(
            index=list(range(len(X))), columns=list(G["dict_node"].keys())
        )
        df_node = df_node.fillna(0)
        df_edge = df_edge.fillna(0)
        df_conf = df_conf.fillna(0)
        for pos_edge_index in range(len(G["list_edge_pos"]) - 1):
            edge_to_analyse = G["list_edge"][
                G["list_edge_pos"][pos_edge_index] : G["list_edge_pos"][
                    pos_edge_index + 1
                ]
            ]
            G_nx = nx.DiGraph(edge_to_analyse)
            degree_to_anaylse = {node: val for (node, val) in G_nx.degree()}
            for edge in edge_to_analyse:
                df_node.at[pos_edge_index, edge[0]] += 1
                df_conf.at[pos_edge_index, edge[0]] = degree_to_anaylse[edge[0]]
                df_edge.at[pos_edge_index, str(edge)] += 1
        return df_node, df_edge, df_conf

    def __create_membership_matrix(self, run):
        mat = np.zeros((len(run), len(run)))
        for i, val_i in enumerate(run):
            for j, val_j in enumerate(run):
                if val_i == val_j:
                    mat[i][j] = 1
                    mat[j][i] = 1
        return mat

    def __build_consensus_matrix(self, all_runs):
        all_mat = sum([self.__create_membership_matrix(run) for run in all_runs])
        return all_mat / all_mat.diagonal()

    def __Clustering_df(self, df):
        Method = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=1)
        df_normalized = df
        df_normalized = df_normalized.subtract(df_normalized.mean(axis=1), axis=0).div(
            df_normalized.std(axis=1), axis=0
        )
        df_normalized = df_normalized.fillna(0)

        kmeans = Method.fit(df_normalized.values)
        return kmeans.labels_

    def __run_clustering_length(self, X, pattern_length):
        G = self.__create_graph(
            X=X,
            length_pattern=max(pattern_length, 4),
            latent=max(1, pattern_length // 3),
            rate=self.rate,
        )
        df_node, df_edge, df_conf = self.__create_dataset(G, X)

        clustering_pred = self.__Clustering_df(
            pd.concat([df_node, df_edge, df_conf], axis=1)
        )

        return [
            clustering_pred,
            G,
            pd.concat([df_node, df_edge], axis=1),
            pattern_length,
        ]

    def __create_graph(self, X, length_pattern, latent, rate):

        dict_result_P = self.__run_proj(X, length_pattern, latent)
        dict_result_G = self.__create_graph_from_proj(
            dict_result_P, rate, length_pattern
        )

        return dict_result_G

    def __create_graph_from_proj(self, dict_result, rate, length_pattern):

        res_point, res_dist = self.__get_intersection_from_radius(
            dict_result["A"], dict_result["index_pos"], rate=rate
        )
        nodes_set, node_weight = self.__nodes_extraction(
            dict_result["A"],
            res_point,
            res_dist,
            rate=rate,
            pattern_length=length_pattern,
        )

        dict_edge_all, dict_node_all = {}, {}
        list_edge_all, edge_in_time_all, list_edge_pos = [], [], []

        for i in range(len(dict_result["index_pos"]) - 1):
            sub_A = dict_result["A"][
                dict_result["index_pos"][i] : dict_result["index_pos"][i + 1]
            ]
            list_edge, edge_in_time, dict_edge, dict_node = self.__edges_extraction(
                sub_A, nodes_set, rate=rate
            )
            list_edge_pos.append(len(list_edge_all))
            list_edge_all += list_edge
            edge_in_time_all.append(edge_in_time)
            dict_edge_all = self.__merge_dict(dict_edge_all, dict_edge)
            dict_node_all = self.__merge_dict(dict_node_all, dict_node)
        list_edge_pos.append(len(list_edge_all))
        return {
            "list_edge": list_edge_all,
            "dict_edge": dict_edge_all,
            "dict_node": dict_node_all,
            "list_edge_pos": list_edge_pos,
            "edge_in_time": edge_in_time_all,
        }

    def __run_proj(self, X, length_pattern, latent):

        if self.variable_length:
            min_X = np.min([np.min(X_sub) for X_sub in X])
            max_X = np.max([np.max(X_sub) for X_sub in X])
        else:
            min_X = np.min(X)
            max_X = np.max(X)
        downsample = max(1, latent // 100)

        X_ref = []
        for i in np.arange(min_X, max_X, (max_X - min_X) / 100):
            tmp = []
            T = [i] * length_pattern
            for j in range(length_pattern - latent):
                tmp.append(sum(x for x in T[j : j + latent]))
            X_ref.append(tmp[::downsample])
        X_ref = np.array(X_ref)

        phase_space_train_list = []
        index_pos = []
        current_length = 0
        for X_sub in X:
            to_add = self.__build_phase_space_smpl(X_sub, latent, length_pattern)
            index_pos.append(current_length)
            phase_space_train_list.append(to_add)
            current_length += len(to_add)

        if len(phase_space_train_list) == 1:
            phase_space_train = phase_space_train_list[0]
        else:
            phase_space_train = np.concatenate(phase_space_train_list, axis=0)

        sample = self.sample  # max(1,latent//2)#4

        pca_1 = PCA(n_components=3, svd_solver="randomized").fit(
            phase_space_train[
                np.random.choice(
                    len(phase_space_train),
                    size=len(phase_space_train) // sample,
                    replace=False,
                )
            ]
        )

        reduced = pca_1.transform(phase_space_train)
        reduced_ref = pca_1.transform(X_ref)

        v_1 = reduced_ref[0]

        R = self.__get_rotation_matrix(v_1, [0.0, 0.0, 1.0])
        A = np.dot(R, reduced.T)
        A = A.T

        return {"pca": pca_1, "A": np.array(A)[:, 0:2], "R": R, "index_pos": index_pos}

    def __build_phase_space_smpl(self, X_sub, latent, m):

        tmp_glob = []
        downsample = max(1, latent // 100)
        current_seq = [0] * m
        first = True
        for i in range(len(X_sub) - m):
            tmp = []
            if first:
                first = False
                for j in range(m - latent):
                    tmp.append(sum(x for x in X_sub[i + j : i + j + latent]))
                tmp_glob.append(tmp[::downsample])
                current_seq = tmp
            else:
                tmp = current_seq[1:]
                tmp.append(sum(x for x in X_sub[i + m - latent : i + m]))
                tmp_glob.append(tmp[::downsample])
                current_seq = tmp

        return np.array(tmp_glob)

    def __get_rotation_matrix(self, i_v, unit):

        curve_vec_1 = i_v
        curve_vec_2 = unit
        a = (curve_vec_1 / np.linalg.norm(curve_vec_1)).reshape(3)
        b = (curve_vec_2 / np.linalg.norm(curve_vec_2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        I = np.identity(3)
        vXStr = "{} {} {}; {} {} {}; {} {} {}".format(
            0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0
        )
        k = np.matrix(vXStr)
        r = I + k + k @ k * ((1 - c) / (s**2))

        return r

    def __find_theta_to_check(self, A, k, rate):
        k_0 = A[k, 0]
        k_1 = A[k, 1]
        k_1_0 = A[k + 1, 0]
        k_1_1 = A[k + 1, 1]
        dist_to_0 = np.sqrt(k_0**2 + k_1**2)
        dist_to_1 = np.sqrt(k_1_0**2 + k_1_1**2)
        theta_point = np.arctan2([k_1 / dist_to_0], [k_0 / dist_to_0])[0]
        theta_point_1 = np.arctan2([k_1_1 / dist_to_1], [k_1_0 / dist_to_1])[0]
        if theta_point < 0:
            theta_point += 2 * np.pi
        if theta_point_1 < 0:
            theta_point_1 += 2 * np.pi
        theta_point = int(theta_point / (2.0 * np.pi) * (rate))
        theta_point_1 = int(theta_point_1 / (2.0 * np.pi) * (rate))
        diff_theta = abs(theta_point - theta_point_1)
        if diff_theta > rate // 2:
            if theta_point_1 > rate // 2:
                diff_theta = abs(theta_point - (-rate + theta_point_1))
            elif theta_point > rate // 2:
                diff_theta = abs((-rate + theta_point) - theta_point_1)
        diff_theta = min(diff_theta, rate // 2)
        theta_to_check = [
            (theta_point + lag) % rate for lag in range(-diff_theta - 1, diff_theta + 1)
        ]

        return theta_to_check

    def __distance(self, a, b):

        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def __det(self, a, b):

        return a[0] * b[1] - a[1] * b[0]

    def __PointsInCircum(self, r, n=500):

        return np.array(
            [
                [math.cos(2 * np.pi / n * x) * r, math.sin(2 * np.pi / n * x) * r]
                for x in range(0, n)
            ]
        )

    def __line_intersection(self, line1, line2):

        xdiff = (line1[0, 0] - line1[1, 0], line2[0, 0] - line2[1, 0])
        ydiff = (line1[0, 1] - line1[1, 1], line2[0, 1] - line2[1, 1])

        div = self.__det(xdiff, ydiff)
        if div == 0:
            return None, None

        max_x_1 = max(line1[0, 0], line1[1, 0])
        max_x_2 = max(line2[0, 0], line2[1, 0])
        max_y_1 = max(line1[0, 1], line1[1, 1])
        max_y_2 = max(line2[0, 1], line2[1, 1])

        min_x_1 = min(line1[0, 0], line1[1, 0])
        min_x_2 = min(line2[0, 0], line2[1, 0])
        min_y_1 = min(line1[0, 1], line1[1, 1])
        min_y_2 = min(line2[0, 1], line2[1, 1])

        d = (self.__det(*line1), self.__det(*line2))
        x = self.__det(d, xdiff) / div
        y = self.__det(d, ydiff) / div
        if not (
            ((x <= max_x_1) and (x >= min_x_1)) and ((x <= max_x_2) and (x >= min_x_2))
        ):
            return None, None
        if not (
            ((y <= max_y_1) and (y >= min_y_1)) and ((y <= max_y_2) and (y >= min_y_2))
        ):
            return None, None

        return [x, y], self.__distance(line1[0], [x, y])

    def __get_intersection_from_radius(self, A, index_pos, rate):

        max_1 = max(max(A[:, 0]), abs(min(A[:, 0])))
        max_2 = max(max(A[:, 1]), abs(min(A[:, 1])))
        set_point = self.__PointsInCircum(np.sqrt(max_1**2 + max_2**2), n=rate)
        previous_node = "not_defined"

        result = [[] for i in range(len(set_point))]
        result_dist = [[] for i in range(len(set_point))]

        for k in random.sample(
            list(range(0, len(A) - 1)), len(list(range(0, len(A) - 1))) // self.sample
        ):
            # if k-1 not in index_pos[1:]:
            theta_to_check = self.__find_theta_to_check(A, k, rate)
            was_found = False
            for i in theta_to_check:
                intersect, dist = self.__line_intersection(
                    np.array([[0, 0], set_point[i]]), np.array([A[k], A[k + 1]])
                )
                if intersect is not None:
                    was_found = True
                    result[i].append(intersect)
                    result_dist[i].append(dist)
                elif (was_found == True) and intersect is None:
                    break
        new_result_dist = []
        for i, res_d in enumerate(result_dist):
            new_result_dist += res_d

        return result, new_result_dist

    def __kde_scipy(self, x, x_grid):

        kde = gaussian_kde(x, bw_method="scott")

        return list(kde.evaluate(x_grid))

    def __nodes_extraction(self, A, res_point, res_dist, rate, pattern_length):

        max_all = max(
            max(max(A[:, 0]), max(A[:, 1])), max(-min(A[:, 0]), -min(A[:, 1]))
        )
        max_all = max_all * 1.2
        range_val_distrib = np.arange(0, max_all, max_all / 250.0)
        list_maxima = []
        list_maxima_val = []
        for segment in range(rate):
            pos_start = sum(len(res_point[i]) for i in range(segment))
            if len(res_dist[pos_start : pos_start + len(res_point[segment])]) == 0:
                self.__verboseprint(
                    "[WARNING] for Graph {}: No intersection found for at least one radius. Sample might be too high.".format(
                        pattern_length
                    )
                )
                maxima_ind = [0]
                maxima_val = [0]
            elif len(res_dist[pos_start : pos_start + len(res_point[segment])]) == 1:
                self.__verboseprint(
                    "[WARNING] for Graph {}: Few intersection found for at least one radius. Sample might be too high.".format(
                        pattern_length
                    )
                )
                maxima_ind = [
                    res_dist[pos_start : pos_start + len(res_point[segment])][0]
                ]
                maxima_val = [1]
            else:
                dist_on_segment = self.__kde_scipy(
                    res_dist[pos_start : pos_start + len(res_point[segment])],
                    range_val_distrib,
                )
                dist_on_segment = (dist_on_segment - min(dist_on_segment)) / (
                    max(dist_on_segment) - min(dist_on_segment)
                )
                maxima = argrelextrema(np.array(dist_on_segment), np.greater)[0]
                if len(maxima) == 0:
                    maxima = np.array([0])
                maxima_ind = [range_val_distrib[val] for val in list(maxima)]
                maxima_val = [dist_on_segment[val] for val in list(maxima)]
            list_maxima.append(maxima_ind)
            list_maxima_val.append(maxima_val)

        return list_maxima, list_maxima_val

    def __find_closest_node(self, list_maxima_ind, point):

        result_list = [np.abs(maxi - point) for maxi in list_maxima_ind]
        result_list_sorted = sorted(result_list)

        return result_list.index(result_list_sorted[0]), result_list_sorted[0]

    def __find_tuple_interseted(self, A, line):

        result = []
        dist_l = []
        for i in range(len(A) - 1):
            intersect, dist = self.__line_intersection(line, np.array([A[i], A[i + 1]]))
            if intersect is not None:
                result.append(intersect)
                dist_l.append(dist)

        return [result, dist_l]

    def __edges_extraction(self, A, set_nodes, rate):

        list_edge = []
        edge_in_time = []
        dict_edge = {}
        dict_node = {}

        max_1 = max(max(A[:, 0]), abs(min(A[:, 0])))
        max_2 = max(max(A[:, 1]), abs(min(A[:, 1])))

        set_point = self.__PointsInCircum(np.sqrt(max_1**2 + max_2**2), n=rate)
        previous_node = "not_defined"

        for k in range(0, len(A) - 1):

            theta_to_check = self.__find_theta_to_check(A, k, rate)
            was_found = False
            for i in theta_to_check:
                to_add = self.__find_tuple_interseted(
                    A[k : k + 2], np.array([[0, 0], set_point[i]])
                )[1]
                if to_add == [] and not was_found:
                    continue
                elif to_add == [] and was_found:
                    break
                else:
                    was_found = True
                    node_in, distance = self.__find_closest_node(
                        set_nodes[i], to_add[0]
                    )

                    if previous_node == "not_defined":
                        previous_node = f"{i}_{node_in}"
                        dict_node[previous_node] = 1

                    else:
                        list_edge.append([previous_node, f"{i}_{node_in}"])

                        if f"{i}_{node_in}" not in dict_node.keys():
                            dict_node[f"{i}_{node_in}"] = 1
                        else:
                            dict_node[f"{i}_{node_in}"] += 1

                        if str(list_edge[-1]) in dict_edge.keys():
                            dict_edge[str(list_edge[-1])] += 1
                        else:
                            dict_edge[str(list_edge[-1])] = 1
                        previous_node = f"{i}_{node_in}"

            edge_in_time.append(len(list_edge))

        return list_edge, edge_in_time, dict_edge, dict_node  # ,list_edge_dist

    def __merge_dict(self, dict_A, dict_B):

        new_dict = {}
        for key in dict_A:
            new_dict[key] = dict_A[key]
        for key in dict_B:
            if key in new_dict.keys():
                new_dict[key] += dict_B[key]
            else:
                new_dict[key] = dict_B[key]

        return new_dict

    # Public methods for visualisation:
    #
    # Our visualisation methods are using Graphviz and pyGraphviz.
    # Some users might have issues installing Graphviz.
    # In case of problems installing PyGraphviz and Graphviz, please
    # remove all functions below

    def show_graphs(
        self, lengths=None, figsize=(30, 40), save_fig=False, namefile=None
    ):
        """
        Plot learned graphs for each length
        """
        """
        Plot learned graphs for each length.

        Parameters
        ----------
        length : list of int or None, optional
            Lengths of graphs to plot (default is None).
            If None, all graphs are plotted.

        savefig: Boolean, optional
            if True, the figure is saved (with name namefile)

        namefile: str or None, optional
            if savefig=True, the figure is saved with the name namefile

        Returns
        -------
        None
        """

        try:
            import pygraphviz
            from networkx.drawing.nx_agraph import graphviz_layout

            graph_viz_used = True
        except ImportError as e:
            print(
                "[WARNING] pygrpaphviz not installed. Please install pygraphviz (and graphviz) for a more approriate graph visualization"
            )
            graph_viz_used = False

        if lengths is None:
            all_lengths = list(self.graphs.keys())
        else:
            all_lengths = lengths

        plt.figure(figsize=figsize)
        for i, length in enumerate(all_lengths):
            plt.subplot(int(len(all_lengths) / 4) + 1, 4, 1 + i)
            self.__plot_graph_length(length, graph_viz_used)
            plt.title(
                "Relevance of length {}: {:.3f}".format(
                    length,
                    self.length_relevance[
                        np.where(self.length_relevance == length)[0], 1
                    ][0],
                )
            )

        if save_fig:
            if namefile is not None:
                plt.savefig(namefile + ".jpg")
            else:
                print("[ERROR]: with save_fig=True, Please provide a namefile")
        else:
            plt.show()
        plt.close()

    def show_graphoids(
        self,
        length=None,
        mode="Exclusive",
        group=False,
        majority_level=0.8,
        figsize=(20, 20),
        save_fig=False,
        namefile=None,
    ):
        """
        Plot the graphoid for a specific length

        Parameters
        ----------
        length : int or None, optional
            Length parameter for graphoid computation (default is None).
            If None, length is set to self.optimal_length.

        mode : {'Raw','Exclusive', 'Proportion'}, optional
            Mode for computing graphoids. If 'Exclusive', compute exclusive graphoids
            (i.e., set to zero the nodes and edges that are not above majority_level).
            If 'Proportion', compute graphoids and normalize the value such that, for one node,
            the sum for all graphoids is equal to one. (default is 'Exclusive').

        majority_level : float, optional
            Majority level threshold for graphoid computation (default is 0.8).
            Not used for Proportion and Raw modes.

        group : Boolean, optional
            if True, Plot each graphoid within an individual plot. Otherwise, plot all
            graphoid within one plot (default is False).

        savefig: Boolean, optional
            if True, the figure is saved (with name namefile)

        namefile: str or None, optional
            if savefig=True, the figure is saved with the name namefile

        Returns
        -------
        None
        """
        try:
            import pygraphviz
            from networkx.drawing.nx_agraph import graphviz_layout

            graph_viz_used = True
        except ImportError as e:
            print(
                "[WARNING] pygrpaphviz not installed. Please install pygraphviz (and graphviz) for a more approriate graph visualization"
            )
            graph_viz_used = False

        if length is None:
            if self.optimal_length is None:
                self.__get_length_relevance()
            length = self.optimal_length

        G_nx = nx.DiGraph(self.graphs[length]["graph"]["list_edge"])
        if graph_viz_used:
            pos = nx.nx_agraph.graphviz_layout(G_nx, prog="fdp")
        else:
            pos = nx.random_layout(G_nx)

        if group:
            plt.figure(figsize=(10, 10))
            self.__plot_graphoid(
                length,
                graphoid="all",
                mode=mode,
                pos=pos,
                majority_level=majority_level,
                graph_viz_used=graph_viz_used,
            )
            plt.title("All graphoids")
        else:
            plt.figure(figsize=figsize)
            for i in range(len(set(self.labels_))):
                plt.subplot(len(set(self.labels_)) // 2 + 1, 2, i + 1)
                self.__plot_graphoid(
                    length,
                    graphoid=i,
                    mode=mode,
                    pos=pos,
                    majority_level=majority_level,
                    graph_viz_used=graph_viz_used,
                )
                plt.title(f"Graph (graphoid in red) for cluster {i}")

        if save_fig:
            if namefile is not None:
                plt.savefig(namefile + ".jpg")
            else:
                print("[ERROR]: with save_fig=True, Please provide a namefile")
        else:
            plt.show()
        plt.close()

    # Private method visualisation

    def __plot_graph_length(self, length, graph_viz_used=False):
        G = self.graphs[length]["graph"]
        G_nx = nx.DiGraph(self.graphs[length]["graph"]["list_edge"])
        if graph_viz_used:
            pos = nx.nx_agraph.graphviz_layout(G_nx, prog="fdp")
        else:
            pos = nx.random_layout(G_nx)
        G_label_0, dict_node_0, edge_size_0 = self.__format_graph_viz(
            G_nx, G["list_edge"], G["dict_node"]
        )
        nx.draw(
            G_label_0, pos=pos, node_size=dict_node_0, linewidths=1, width=edge_size_0
        )
        nx.draw_networkx_labels(G_label_0, pos, font_size=10)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("black")

    def __format_graph_viz(self, G, list_edge, node_weight):
        edge_size = []

        for edge in G.edges():
            edge_size.append(list_edge.count([edge[0], edge[1]]))
        edge_size_b = [
            float(1 + (e - min(edge_size))) / float(1 + max(edge_size) - min(edge_size))
            for e in edge_size
        ]
        edge_size = [min(e * 10, 5) for e in edge_size_b]
        dict_node = []
        for node in G.nodes():
            if node != "NULL_NODE":
                dict_node.append(max(100, node_weight[node]))
            else:
                dict_node.append(100)

        return G, dict_node, edge_size

    def __plot_graphoid(
        self,
        length,
        graphoid="all",
        mode="Exclusive",
        pos=None,
        majority_level=0.8,
        graph_viz_used=False,
    ):

        color_class = LIST_COLOR

        G = self.graphs[length]["graph"]
        G_nx = nx.DiGraph(G["list_edge"])
        if pos is None:
            if graph_viz_used:
                pos = nx.nx_agraph.graphviz_layout(G_nx, prog="fdp")
            else:
                pos = nx.random_layout(G_nx)

        G_label_0, dict_node_0, edge_size_0 = self.__format_graph_viz(
            G_nx, G["list_edge"], G["dict_node"]
        )

        all_graphoid, names_features = self.compute_graphoids(
            length, mode, majority_level
        )
        all_graphoid = np.array(all_graphoid)
        color_map = []
        node_width = []
        labels_text = {}
        for node in G_label_0:
            if graphoid == "all":
                color_n = self.__combine_hex_values(
                    {
                        hc[1:]: val
                        for hc, val in zip(
                            color_class[: len(all_graphoid)],
                            all_graphoid[:, names_features.index(node)],
                        )
                    }
                )
                color_n = self.__combine_hex_values({color_n: 0.95, "a2a2a2": 0.1})
                node_width_val = 1
                labels_text[node] = node
            else:
                color_n = self.__combine_hex_values(
                    {
                        "ff0000": all_graphoid[:, names_features.index(node)][graphoid],
                        "f8f8f8": 0.1,
                    }
                )

                if all_graphoid[:, names_features.index(node)][graphoid] > 0:
                    node_width_val = 1
                    labels_text[node] = node
                else:
                    node_width_val = 0.2

            node_width.append(node_width_val)
            color_map.append("#" + color_n)

        color_map_edge = []
        for i, (u, v) in enumerate(G_label_0.edges()):
            if graphoid == "all":
                color_n = self.__combine_hex_values(
                    {
                        hc[1:]: val
                        for hc, val in zip(
                            color_class[: len(all_graphoid)],
                            all_graphoid[:, names_features.index(f"['{u}', '{v}']")],
                        )
                    }
                )
                color_n = self.__combine_hex_values({color_n: 0.90, "a2a2a2": 0.1})
                edge_size_0[i] = edge_size_0[i] * 2
            else:
                color_n = self.__combine_hex_values(
                    {
                        "ff0000": all_graphoid[
                            :, names_features.index(f"['{u}', '{v}']")
                        ][graphoid],
                        "a2a2a2": 0.1,
                    }
                )
                if (
                    all_graphoid[:, names_features.index(f"['{u}', '{v}']")][graphoid]
                    > 0
                ):
                    edge_size_0[i] = edge_size_0[i] + 5
                else:
                    edge_size_0[i] = 0.5

            color_map_edge.append("#" + color_n)

        nx.draw(
            G_label_0,
            pos=pos,
            node_color=color_map,
            edge_color=color_map_edge,
            node_size=dict_node_0,
            linewidths=node_width,
            width=edge_size_0,
        )
        nx.draw_networkx_labels(G_label_0, pos, labels_text, font_size=10)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("black")

    def __verboseprint(self, *args):
        if self.verbose:
            for arg in args:
                print(arg, end=" ")
            print()
        else:
            verboseprint = lambda *a: None

    def __combine_hex_values(self, d):
        d_items = sorted(d.items())
        tot_weight = sum(d.values())
        zpad = lambda x: x if len(x) == 2 else "0" + x
        if tot_weight == 0:
            return zpad(hex(255)[2:]) + zpad(hex(255)[2:]) + zpad(hex(255)[2:])
        red = int(sum([int(k[:2], 16) * v for k, v in d_items]) / tot_weight)
        green = int(sum([int(k[2:4], 16) * v for k, v in d_items]) / tot_weight)
        blue = int(sum([int(k[4:6], 16) * v for k, v in d_items]) / tot_weight)

        return zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])
