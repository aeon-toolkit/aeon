"""Shapeleter transformation."""

__maintainer__ = []
__all__ = ["ShapeleterTransformer"]

import numpy as np
from numba import njit, prange
from scipy.sparse import lil_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)


@njit(fastmath=True, cache=True)
def _assign_order(positions_abs):
    """Convert absolute shapelet positions to ordinal positions with tie handling."""
    n = len(positions_abs)
    positions_ord = np.zeros(n, dtype=np.float64)
    order = np.argsort(positions_abs)
    i = 0
    while i < n:
        j = i + 1
        val = positions_abs[order[i]]
        while j < n and positions_abs[order[j]] == val:
            j += 1
        curr_n = float(j - i)
        rank_val = float(i) + (curr_n - 1.0) / 2.0
        for k in range(i, j):
            positions_ord[order[k]] = rank_val
        i = j
    return positions_ord


@njit(parallel=True, fastmath=True, cache=True)
def _assign_order_batch(positions_abs_2d):
    """Compute ordinal positions across all cases in parallel."""
    n_cases, n_feats = positions_abs_2d.shape
    out = np.zeros((n_cases, n_feats), dtype=np.float64)
    for c in prange(n_cases):
        out[c] = _assign_order(positions_abs_2d[c])
    return out


def _compute_jaccard_matrix(edges, n_nodes):
    """Compute Jaccard similarity matrix across hypergraph nodes."""
    if len(edges) == 0 or n_nodes == 0:
        return np.eye(max(1, n_nodes))

    num_edges = len(edges)
    h_mat = lil_matrix((n_nodes, num_edges), dtype=int)
    for edge_id, edge in enumerate(edges):
        for node in edge:
            if node < n_nodes:
                h_mat[node, edge_id] = 1

    h_csr = h_mat.tocsr()
    intersection = h_csr.dot(h_csr.T).toarray()
    degrees = np.array(h_csr.sum(axis=1)).flatten()
    union = degrees[:, np.newaxis] + degrees[np.newaxis, :] - intersection

    j_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    np.divide(intersection, union, where=(union != 0), out=j_matrix)
    np.fill_diagonal(j_matrix, 1.0)
    return j_matrix


def _hypergraph_prune(
    shapelets,
    shapelet_len,
    labels,
    neighbor_k=(3, 5),
    save_rate=0.5,
    jaccard_threshold=0.9,
):
    """Prune candidate shapelets using hypergraph purity and Jaccard diversity."""
    n_shapelets = len(labels)
    valid_k = [k for k in neighbor_k if k <= n_shapelets]
    if len(valid_k) == 0 or n_shapelets <= 1:
        return np.arange(n_shapelets)

    # Slice strictly to the known shapelet length to eliminate inf padding
    raw_sub = shapelets[:, 0, :shapelet_len]
    cleaned_sub = np.nan_to_num(raw_sub, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Hyperedges based on nearest shapelet Euclidean distances
    dist_mat = euclidean_distances(cleaned_sub)
    np.fill_diagonal(dist_mat, np.inf)

    edges = []
    for k in valid_k:
        for i in range(n_shapelets):
            edges.append(np.argsort(dist_mat[i])[:k])

    # 2. Node purity calculation
    purity = np.zeros(n_shapelets)
    degree = np.zeros(n_shapelets)
    for edge in edges:
        sub_labels = labels[edge]
        edge_sz = len(edge)
        for idx in edge:
            degree[idx] += 1
            purity[idx] += np.sum(labels[idx] == sub_labels) / edge_sz

    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where(degree > 0, purity / degree, 0.0)

    # 3. Jaccard similarity matrix
    j_matrix = _compute_jaccard_matrix(edges, n_shapelets)
    j_matrix_bin = np.where(j_matrix >= jaccard_threshold, 1, 0)
    np.fill_diagonal(j_matrix_bin, -1)

    # 4. Greedy diversity selection
    save_n = max(1, int(np.ceil(save_rate * n_shapelets)))
    unique_labels = np.unique(labels)
    top_node = int(np.argmax(purity))
    saved_nodes = [top_node]

    lbl_idx = (np.where(unique_labels == labels[top_node])[0][0] + 1) % len(
        unique_labels
    )
    candidate_nodes = [node for node in np.argsort(purity)[::-1] if node != top_node]

    for _ in range(save_n - 1):
        if not candidate_nodes:
            break
        last_saved = saved_nodes[-1]
        similar_nodes = set(np.where(j_matrix_bin[last_saved, :] == 1)[0])
        candidate_nodes = [c for c in candidate_nodes if c not in similar_nodes]
        if not candidate_nodes:
            break

        cand_labels = labels[candidate_nodes]
        target_lbl = unique_labels[lbl_idx]
        if np.sum(cand_labels == target_lbl) == 0:
            lbl_idx = (lbl_idx + 1) % len(unique_labels)
            target_lbl = unique_labels[lbl_idx]

        sub_cands = [c for c in candidate_nodes if labels[c] == target_lbl]
        if not sub_cands:
            sub_cands = candidate_nodes

        best_cand = sub_cands[int(np.argmax(purity[sub_cands]))]
        saved_nodes.append(best_cand)
        candidate_nodes.remove(best_cand)
        lbl_idx = (lbl_idx + 1) % len(unique_labels)

    return np.array(saved_nodes, dtype=int)


class _CombinationFusion:
    """Pairwise LDA fusion of original, absolute, and ordinal views."""

    def __init__(self, threshold=20.0):
        self.threshold = threshold
        self.lda_oa = []
        self.lda_or = []
        self.lda_ar = []
        self.scaler_oa = MinMaxScaler((-1, 1))
        self.scaler_or = MinMaxScaler((-1, 1))
        self.scaler_ar = MinMaxScaler((-1, 1))

    def _fit_single_pair(self, v1, v2, y):
        feat = np.hstack([v1, v2])
        if len(np.unique(y)) > 1:
            try:
                lda = LinearDiscriminantAnalysis(n_components=1)
                lda.fit(feat, y)
                trans = lda.transform(feat)
                return lda, trans
            except Exception:
                pass
        return None, np.mean(feat, axis=1, keepdims=True)

    def _transform_single_pair(self, v1, v2, lda):
        feat = np.hstack([v1, v2])
        if lda is not None:
            try:
                return lda.transform(feat)
            except Exception:
                pass
        return np.mean(feat, axis=1, keepdims=True)

    def fit_transform(self, x_orig, x_abs, x_ord, y):
        n_features = x_orig.shape[1]
        out_oa = np.zeros_like(x_orig)
        out_or = np.zeros_like(x_orig)
        out_ar = np.zeros_like(x_orig)

        self.lda_oa = []
        self.lda_or = []
        self.lda_ar = []

        for i in range(n_features):
            lda, trans = self._fit_single_pair(
                x_orig[:, i : i + 1], x_abs[:, i : i + 1], y
            )
            self.lda_oa.append(lda)
            out_oa[:, i : i + 1] = trans

            lda, trans = self._fit_single_pair(
                x_orig[:, i : i + 1], x_ord[:, i : i + 1], y
            )
            self.lda_or.append(lda)
            out_or[:, i : i + 1] = trans

            lda, trans = self._fit_single_pair(
                x_abs[:, i : i + 1], x_ord[:, i : i + 1], y
            )
            self.lda_ar.append(lda)
            out_ar[:, i : i + 1] = trans

        np.clip(out_oa, -self.threshold, self.threshold, out=out_oa)
        np.clip(out_or, -self.threshold, self.threshold, out=out_or)
        np.clip(out_ar, -self.threshold, self.threshold, out=out_ar)

        return (
            self.scaler_oa.fit_transform(out_oa),
            self.scaler_or.fit_transform(out_or),
            self.scaler_ar.fit_transform(out_ar),
        )

    def transform(self, x_orig, x_abs, x_ord):
        n_features = x_orig.shape[1]
        out_oa = np.zeros_like(x_orig)
        out_or = np.zeros_like(x_orig)
        out_ar = np.zeros_like(x_orig)

        for i in range(n_features):
            out_oa[:, i : i + 1] = self._transform_single_pair(
                x_orig[:, i : i + 1], x_abs[:, i : i + 1], self.lda_oa[i]
            )
            out_or[:, i : i + 1] = self._transform_single_pair(
                x_orig[:, i : i + 1], x_ord[:, i : i + 1], self.lda_or[i]
            )
            out_ar[:, i : i + 1] = self._transform_single_pair(
                x_abs[:, i : i + 1], x_ord[:, i : i + 1], self.lda_ar[i]
            )

        np.clip(out_oa, -self.threshold, self.threshold, out=out_oa)
        np.clip(out_or, -self.threshold, self.threshold, out=out_or)
        np.clip(out_ar, -self.threshold, self.threshold, out=out_ar)

        return (
            self.scaler_oa.transform(out_oa),
            self.scaler_or.transform(out_or),
            self.scaler_ar.transform(out_ar),
        )


class ShapeleterTransformer(BaseCollectionTransformer):
    """Shapeleter transformer with hypergraph pruning and dual positional embedding.

    Parameters
    ----------
    max_shapelets : int, default=1000
        Maximum number of shapelets sampled initially by RDST backbone.
    ka : float, default=1.5
        Absolute positional scaling parameter.
    ko : float, default=1.5
        Ordinal positional scaling parameter.
    save_rate : float, default=0.5
        Proportion of shapelets preserved by hypergraph selection.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness.
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        max_shapelets=1000,
        ka=1.5,
        ko=1.5,
        save_rate=0.5,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.ka = ka
        self.ko = ko
        self.save_rate = save_rate
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit RDST backbone, apply hypergraph pruning, and initialize fusions."""
        if y is None:
            y = np.zeros(X.shape[0], dtype=int)

        self._series_len = X.shape[-1]
        self._rdst = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            random_state=self.random_state,
        )
        self._rdst.fit(X, y)

        raw_shapelets = self._rdst.shapelets_[0]
        n_candidates = len(raw_shapelets)

        # Multi-scale hypergraph pruning grouped by (length, dilation, normalization)
        if n_candidates > 0 and len(np.unique(y)) > 1:
            lengths = self._rdst.shapelets_[2]
            dilas = self._rdst.shapelets_[3]
            norms = self._rdst.shapelets_[5]
            selected_idx = []

            for l_val in np.unique(lengths):
                for d_val in np.unique(dilas):
                    for norm_flag in [0, 1]:
                        mask = (
                            (lengths == l_val) & (dilas == d_val) & (norms == norm_flag)
                        )
                        sub_idx = np.where(mask)[0]
                        if len(sub_idx) > 0:
                            pruned_rel = _hypergraph_prune(
                                raw_shapelets[sub_idx],
                                l_val,
                                y[sub_idx % len(y)],
                                save_rate=self.save_rate,
                            )
                            selected_idx.append(sub_idx[pruned_rel])

            if len(selected_idx) > 0:
                selected_idx = np.hstack(selected_idx)
                self._rdst.shapelets_ = tuple(
                    attr[selected_idx] for attr in self._rdst.shapelets_
                )

        self._shapelet_lengths = self._rdst.shapelets_[2]
        self._shapelet_dilations = self._rdst.shapelets_[3]

        # Extract features to initialize scalers & fusion
        raw_feats = self._rdst.transform(X)
        raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale SOO features
        n_shapelets = len(self._shapelet_lengths)
        for i in range(n_shapelets):
            denom = (
                self._series_len
                - (self._shapelet_lengths[i] - 1) * self._shapelet_dilations[i]
            )
            denom = max(1.0, float(denom))
            raw_feats[:, 2 + 3 * i] = raw_feats[:, 2 + 3 * i] / denom

        x_min = raw_feats[:, 0::3]
        x_arg = raw_feats[:, 1::3]
        x_soo = raw_feats[:, 2::3]

        self._scaler_min = MinMaxScaler((-1, 1)).fit(x_min)
        self._scaler_soo = MinMaxScaler((-1, 1)).fit(x_soo)
        x_min_scaled = self._scaler_min.transform(x_min)
        x_soo_scaled = self._scaler_soo.transform(x_soo)

        x_min_abs, x_min_ord, x_soo_abs, x_soo_ord = self._encode_positions(
            x_min_scaled, x_soo_scaled, x_arg
        )

        self._fusion_min = _CombinationFusion()
        self._fusion_soo = _CombinationFusion()

        self._fusion_min.fit_transform(x_min_scaled, x_min_abs, x_min_ord, y)
        self._fusion_soo.fit_transform(x_soo_scaled, x_soo_abs, x_soo_ord, y)
        return self

    def _encode_positions(self, x_min, x_soo, x_arg):
        """Dual absolute and ordinal sinusoidal positional encoding."""
        x_min_abs = np.copy(x_min)
        x_min_ord = np.copy(x_min)
        x_soo_abs = np.copy(x_soo)
        x_soo_ord = np.copy(x_soo)

        for d in np.unique(self._shapelet_dilations):
            sub_idx = np.where(self._shapelet_dilations == d)[0]
            sh_len = self._shapelet_lengths[sub_idx[0]]
            len_map = max(1.0, float(self._series_len - (sh_len - 1) * d))
            n_sub = max(1.0, float(len(sub_idx)))

            base_abs = self.ka * len_map / (2.0 * np.pi)
            base_ord = self.ko * n_sub / (2.0 * np.pi)

            # Absolute
            pos_abs = x_arg[:, sub_idx]
            x_min_abs[:, sub_idx] += np.sin(pos_abs / base_abs)
            x_soo_abs[:, sub_idx] += np.sin(pos_abs / base_abs)

            # Ordinal with tie-break
            pos_ord = _assign_order_batch(pos_abs)

            x_min_ord[:, sub_idx] += np.sin(pos_ord / base_ord)
            x_soo_ord[:, sub_idx] += np.sin(pos_ord / base_ord)

        return x_min_abs, x_min_ord, x_soo_abs, x_soo_ord

    def _transform(self, X, y=None):
        """Transform time series into concatenated multi-view shapelet features."""
        raw_feats = self._rdst.transform(X)
        raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

        n_shapelets = len(self._shapelet_lengths)
        for i in range(n_shapelets):
            denom = (
                self._series_len
                - (self._shapelet_lengths[i] - 1) * self._shapelet_dilations[i]
            )
            denom = max(1.0, float(denom))
            raw_feats[:, 2 + 3 * i] = raw_feats[:, 2 + 3 * i] / denom

        x_min = self._scaler_min.transform(raw_feats[:, 0::3])
        x_soo = self._scaler_soo.transform(raw_feats[:, 2::3])
        x_arg = raw_feats[:, 1::3]

        x_min_abs, x_min_ord, x_soo_abs, x_soo_ord = self._encode_positions(
            x_min, x_soo, x_arg
        )

        m_oa, m_or, m_ar = self._fusion_min.transform(x_min, x_min_abs, x_min_ord)
        s_oa, s_or, s_ar = self._fusion_soo.transform(x_soo, x_soo_abs, x_soo_ord)

        # Concatenate all 8 multiviews
        return np.hstack([x_min, x_soo, m_oa, m_or, m_ar, s_oa, s_or, s_ar])
