"""Shapeleter transformation."""

__maintainer__ = []
__all__ = [
    "ShapeleterTransform",
    "_hilbert_transform",
    "_series_transform",
    "_cosine_similarity_squared_loss",
]

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
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
    edge_lens = [len(e) for e in edges]
    cols = np.repeat(np.arange(num_edges), edge_lens)
    rows = np.concatenate(edges)
    data = np.ones(len(rows), dtype=np.int32)

    h_csr = csr_matrix((data, (rows, cols)), shape=(n_nodes, num_edges))
    intersection = (h_csr @ h_csr.T).toarray()
    degrees = np.asarray(h_csr.sum(axis=1)).ravel()
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
    per_shapelets=None,
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

    # 1. Hyperedges based on nearest shapelet Euclidean distances (including self)
    dist_mat = euclidean_distances(cleaned_sub)
    np.fill_diagonal(dist_mat, 0.0)

    edges = []
    for k in valid_k:
        for i in range(n_shapelets):
            nn_indices = np.argsort(dist_mat[i])[:k]
            if i not in nn_indices:
                nn_indices[-1] = i
            edges.append(nn_indices)

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
    if per_shapelets is not None:
        save_n = min(save_n, int(per_shapelets))
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


def _cosine_similarity_squared_loss(f0, f1, eps=1e-8):
    """Compute mean squared cosine similarity across corresponding columns."""
    dot = np.sum(f0 * f1, axis=0)
    norm0 = np.sqrt(np.sum(f0**2, axis=0))
    norm1 = np.sqrt(np.sum(f1**2, axis=0))
    denom = np.maximum(norm0 * norm1, eps)
    cos_sim = dot / denom
    return np.mean(cos_sim**2)


def _compute_pe_loss(w, featureX, positionX, orderX, len_map, k_min=1.0, k_max=15.0):
    """Compute cosine similarity squared loss for given log_Ka and log_Ko."""
    log_ka = np.clip(w[0], np.log(k_min), np.log(k_max))
    log_ko = np.clip(w[1], np.log(k_min), np.log(k_max))
    ka = np.exp(log_ka)
    ko = np.exp(log_ko)

    n_shapelets = positionX.shape[1]
    base_abs = ka * len_map / (2.0 * np.pi)
    base_ord = ko * n_shapelets / (2.0 * np.pi)

    pe_abs = np.hstack([np.sin(positionX / base_abs), np.cos(positionX / base_abs)])
    pe_ord = np.hstack([np.sin(orderX / base_ord), np.cos(orderX / base_ord)])

    f_a = featureX + pe_abs
    f_o = featureX + pe_ord

    cos_ra = _cosine_similarity_squared_loss(featureX, f_a)
    cos_ro = _cosine_similarity_squared_loss(featureX, f_o)
    cos_ao = _cosine_similarity_squared_loss(f_a, f_o)
    return -(cos_ra + cos_ro + cos_ao) / 3.0


def _optimize_position_encoding_np(
    featureX,
    positionX,
    orderX,
    len_map,
    ka0=8.0,
    ko0=8.0,
    k_min=1.0,
    k_max=15.0,
    lr=0.002,
    max_iter=200,
):
    """Optimize Ka and Ko using Adam in pure NumPy."""
    w = np.array([np.log(ka0), np.log(ko0)], dtype=np.float64)
    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    h = 1e-5

    for t in range(1, max_iter + 1):
        grad = np.zeros(2, dtype=np.float64)
        for i in range(2):
            w_plus = w.copy()
            w_plus[i] += h
            w_minus = w.copy()
            w_minus[i] -= h
            l_plus = _compute_pe_loss(
                w_plus, featureX, positionX, orderX, len_map, k_min, k_max
            )
            l_minus = _compute_pe_loss(
                w_minus, featureX, positionX, orderX, len_map, k_min, k_max
            )
            grad[i] = (l_plus - l_minus) / (2.0 * h)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        w[0] = np.clip(w[0], np.log(k_min), np.log(k_max))
        w[1] = np.clip(w[1], np.log(k_min), np.log(k_max))

    return float(np.exp(w[0])), float(np.exp(w[1]))


def _hilbert_transform(signal):
    """Compute discrete Hilbert transform (envelope of analytic signal) via FFT.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array with time along the last axis.

    Returns
    -------
    envelope : np.ndarray
        Analytic signal envelope (magnitude).
    """
    n = signal.shape[-1]
    if n <= 1:
        return np.abs(signal)

    f = np.fft.fft(signal, axis=-1)
    h = np.zeros(n, dtype=np.float64)
    h[0] = 1.0
    if n % 2 == 0:
        h[1 : n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[1 : (n + 1) // 2] = 2.0

    analytic = np.fft.ifft(f * h, axis=-1)
    return np.abs(analytic)


def _series_transform(X, mode):
    """Transform time series collection into alternative representation.

    Parameters
    ----------
    X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
        Input time series collection.
    mode : {"F", "H", "HF"}
        Transformation mode:
        - "F": First-order difference series.
        - "H": Hilbert transform (amplitude envelope).
        - "HF": Hilbert transform of first-order difference series.

    Returns
    -------
    X_trans : np.ndarray
        Z-normalized transformed time series collection.
    """
    if mode == "F":
        s = np.diff(X, axis=-1)
    elif mode == "H":
        s = _hilbert_transform(X)
    elif mode == "HF":
        s = _hilbert_transform(np.diff(X, axis=-1))
    else:
        raise ValueError(
            f"Unknown transformation mode: {mode}. Expected 'F', 'H', or 'HF'."
        )

    mean = np.mean(s, axis=-1, keepdims=True)
    std = np.std(s, axis=-1, keepdims=True)
    return (s - mean) / (std + 1e-8)


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
                if trans.ndim == 2 and trans.shape[1] == 1:
                    return lda, trans
            except Exception:
                pass
        return None, np.mean(feat, axis=1, keepdims=True)

    def _transform_single_pair(self, v1, v2, lda):
        feat = np.hstack([v1, v2])
        if lda is not None:
            try:
                trans = lda.transform(feat)
                if trans.ndim == 2 and trans.shape[1] == 1:
                    return trans
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


class ShapeleterTransform(BaseCollectionTransformer):
    """Shapeleter transformer with hypergraph pruning and dual positional embedding.

    Shapeleter [1]_ enhances shapelet-based representations by combining a random
    dilated shapelet transform (RDST) backbone with multi-scale hypergraph pruning
    and dual positional embeddings (absolute positions and ordinal ranks).

    The transformation pipeline consists of four main phases:
    1. **Candidate Shapelet Extraction**: Random shapelet candidates are extracted
       using the RandomDilatedShapeletTransform (RDST) backbone.
    2. **Multi-Scale Hypergraph Pruning**: Candidates are grouped by length, dilation,
       and normalization. For each group, a hypergraph is constructed based on
       nearest-neighbor distances (including the shapelet itself). Shapelets are
       ranked by neighborhood purity and filtered greedily using Jaccard similarity
       to retain informative, diverse shapelets.
    3. **Dual Positional Embeddings**: For each preserved shapelet, both absolute
       temporal occurrence positions (argmin) and tie-aware ordinal ranks are
       encoded via sinusoidal transformations. Scaling parameters :math:`K_a`
       (absolute) and :math:`K_o` (ordinal) are optimized per dilation using Adam
       on a cosine-similarity loss function.
    4. **Multi-View Fusion**: Original features and position-enhanced features are
       fused using pairwise Linear Discriminant Analysis (LDA), producing 8
       complementary feature views.

    Parameters
    ----------
    max_shapelets : int, default=1000
        Maximum number of candidate shapelets sampled initially by the RDST backbone.
    per_shapelets : int or None, default=None
        Maximum number of shapelets preserved per group after hypergraph pruning.
        If None, selection is governed solely by `save_rate`.
    ka : float, default=8.0
        Initial absolute positional scaling parameter (:math:`K_{a0}`).
    ko : float, default=8.0
        Initial ordinal positional scaling parameter (:math:`K_{o0}`).
    k_min : float, default=1.0
        Lower bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    k_max : float, default=15.0
        Upper bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    lr : float, default=0.002
        Learning rate for Adam optimization of :math:`K_a` and :math:`K_o`.
    max_iter : int, default=200
        Maximum number of iterations for Adam positional optimization per dilation.
        If set to 0, optimization is skipped and fixed `ka` and `ko` values are used.
    save_rate : float, default=0.5
        Proportion of candidate shapelets preserved by hypergraph selection in each
        (length, dilation, normalization) group.
    random_state : int, RandomState instance or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    rdst_ : RandomDilatedShapeletTransform
        The underlying fitted RDST shapelet backbone.
    shapelet_lengths_ : np.ndarray
        Array of shapelet lengths for all preserved shapelets.
    shapelet_dilations_ : np.ndarray
        Array of shapelet dilations for all preserved shapelets.
    series_len_ : int
        Length of the input time series.
    unique_dilations_ : np.ndarray
        Array of unique dilations among preserved shapelets.
    ka_opt_ : dict
        Mapping from dilation to optimized absolute positional scaling factor Ka.
    ko_opt_ : dict
        Mapping from dilation to optimized ordinal positional scaling factor Ko.
    base_abs_ : dict
        Mapping from dilation to optimized absolute positional base scaling factor.
    base_ord_ : dict
        Mapping from dilation to optimized ordinal positional base scaling factor.
    scaler_min_ : MinMaxScaler
        Fitted scaler for minimum shapelet distance features.
    scaler_soo_ : MinMaxScaler
        Fitted scaler for shapelet occurrence (SOO) features.
    fusion_min_ : _CombinationFusion
        Fitted pairwise LDA fusion model for minimum distance views.
    fusion_soo_ : _CombinationFusion
        Fitted pairwise LDA fusion model for SOO views.

    References
    ----------
    .. [1] Chang-Cheng He et al. "Shapeleter: Shapelet enhancer via multiview
       positional embedding for time series classification and extrinsic regression",
       Information Fusion, 2024.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.shapelet_based import ShapeleterTransform
    >>> X = np.random.RandomState(42).normal(size=(10, 1, 50))
    >>> y = np.array([0, 1] * 5)
    >>> st = ShapeleterTransform(max_shapelets=50, max_iter=5, random_state=42)
    >>> Xt = st.fit_transform(X, y)
    >>> Xt.shape[0] == 10
    True
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
        "python_dependencies": None,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        max_shapelets=1000,
        per_shapelets=None,
        ka=8.0,
        ko=8.0,
        k_min=1.0,
        k_max=15.0,
        lr=0.002,
        max_iter=200,
        save_rate=0.5,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.per_shapelets = per_shapelets
        self.ka = ka
        self.ko = ko
        self.k_min = k_min
        self.k_max = k_max
        self.lr = lr
        self.max_iter = max_iter
        self.save_rate = save_rate
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Fit RDST backbone, apply hypergraph pruning, and initialize fusions."""
        if y is None:
            y = np.zeros(X.shape[0], dtype=int)

        self.series_len_ = X.shape[-1]
        self.rdst_ = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            random_state=self.random_state,
        )
        self.rdst_.fit(X, y)

        raw_shapelets = self.rdst_.shapelets_[0]
        n_candidates = len(raw_shapelets)

        # Multi-scale hypergraph pruning grouped by (length, dilation, normalization)
        if n_candidates > 0 and len(np.unique(y)) > 1:
            lengths = self.rdst_.shapelets_[2]
            dilas = self.rdst_.shapelets_[3]
            norms = self.rdst_.shapelets_[5]
            shapelet_classes = self.rdst_.shapelets_[8]
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
                                shapelet_classes[sub_idx],
                                save_rate=self.save_rate,
                                per_shapelets=self.per_shapelets,
                            )
                            selected_idx.append(sub_idx[pruned_rel])

            if len(selected_idx) > 0:
                selected_idx = np.hstack(selected_idx)
                self.rdst_.shapelets_ = tuple(
                    attr[selected_idx] for attr in self.rdst_.shapelets_
                )

        self.shapelet_lengths_ = self.rdst_.shapelets_[2]
        self.shapelet_dilations_ = self.rdst_.shapelets_[3]

        # Extract features to initialize scalers & fusion
        raw_feats = self.rdst_.transform(X)
        raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale SOO features
        n_shapelets = len(self.shapelet_lengths_)
        for i in range(n_shapelets):
            denom = (
                self.series_len_
                - (self.shapelet_lengths_[i] - 1) * self.shapelet_dilations_[i]
            )
            denom = max(1.0, float(denom))
            raw_feats[:, 2 + 3 * i] = raw_feats[:, 2 + 3 * i] / denom

        x_min = raw_feats[:, 0::3]
        x_arg = raw_feats[:, 1::3]
        x_soo = raw_feats[:, 2::3]

        self.scaler_min_ = MinMaxScaler((-1, 1)).fit(x_min)
        self.scaler_soo_ = MinMaxScaler((-1, 1)).fit(x_soo)
        x_min_scaled = self.scaler_min_.transform(x_min)
        x_soo_scaled = self.scaler_soo_.transform(x_soo)

        self.unique_dilations_ = np.unique(self.shapelet_dilations_)
        self._unique_dilations = self.unique_dilations_
        self.base_abs_ = {}
        self.base_ord_ = {}
        self.ka_opt_ = {}
        self.ko_opt_ = {}

        for d in self.unique_dilations_:
            sub_idx = np.where(self.shapelet_dilations_ == d)[0]
            sh_len = self.shapelet_lengths_[sub_idx[0]]
            len_map = max(1.0, float(self.series_len_ - (sh_len - 1) * d))
            n_sub = max(1.0, float(len(sub_idx)))

            if self.max_iter > 0:
                train_feat_d = np.hstack(
                    [x_min_scaled[:, sub_idx], x_soo_scaled[:, sub_idx]]
                )
                pos_abs_d = x_arg[:, sub_idx]
                pos_ord_d = _assign_order_batch(pos_abs_d)

                ka_d, ko_d = _optimize_position_encoding_np(
                    train_feat_d,
                    pos_abs_d,
                    pos_ord_d,
                    len_map,
                    ka0=self.ka,
                    ko0=self.ko,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    lr=self.lr,
                    max_iter=self.max_iter,
                )
            else:
                ka_d, ko_d = self.ka, self.ko

            self.ka_opt_[d] = ka_d
            self.ko_opt_[d] = ko_d
            self.base_abs_[d] = ka_d * len_map / (2.0 * np.pi)
            self.base_ord_[d] = ko_d * n_sub / (2.0 * np.pi)

        self._base_abs = self.base_abs_
        self._base_ord = self.base_ord_

        x_min_abs, x_min_ord, x_soo_abs, x_soo_ord = self._encode_positions(
            x_min_scaled, x_soo_scaled, x_arg
        )

        self.fusion_min_ = _CombinationFusion()
        self.fusion_soo_ = _CombinationFusion()

        self.fusion_min_.fit_transform(x_min_scaled, x_min_abs, x_min_ord, y)
        self.fusion_soo_.fit_transform(x_soo_scaled, x_soo_abs, x_soo_ord, y)
        return self

    def _encode_positions(self, x_min, x_soo, x_arg):
        """Dual absolute and ordinal sinusoidal positional encoding."""
        x_min_abs = np.copy(x_min)
        x_min_ord = np.copy(x_min)
        x_soo_abs = np.copy(x_soo)
        x_soo_ord = np.copy(x_soo)

        for d in self.unique_dilations_:
            sub_idx = np.where(self.shapelet_dilations_ == d)[0]
            base_abs = self.base_abs_[d]
            base_ord = self.base_ord_[d]

            # Absolute
            pos_abs = x_arg[:, sub_idx]
            x_min_abs[:, sub_idx] += np.sin(pos_abs / base_abs)
            x_soo_abs[:, sub_idx] += np.cos(pos_abs / base_abs)

            # Ordinal with tie-break
            pos_ord = _assign_order_batch(pos_abs)

            x_min_ord[:, sub_idx] += np.sin(pos_ord / base_ord)
            x_soo_ord[:, sub_idx] += np.cos(pos_ord / base_ord)

        return x_min_abs, x_min_ord, x_soo_abs, x_soo_ord

    def _transform(self, X, y=None):
        """Transform time series into concatenated multi-view shapelet features."""
        raw_feats = self.rdst_.transform(X)
        raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

        n_shapelets = len(self.shapelet_lengths_)
        for i in range(n_shapelets):
            denom = (
                self.series_len_
                - (self.shapelet_lengths_[i] - 1) * self.shapelet_dilations_[i]
            )
            denom = max(1.0, float(denom))
            raw_feats[:, 2 + 3 * i] = raw_feats[:, 2 + 3 * i] / denom

        x_min = self.scaler_min_.transform(raw_feats[:, 0::3])
        x_soo = self.scaler_soo_.transform(raw_feats[:, 2::3])
        x_arg = raw_feats[:, 1::3]

        x_min_abs, x_min_ord, x_soo_abs, x_soo_ord = self._encode_positions(
            x_min, x_soo, x_arg
        )

        m_oa, m_or, m_ar = self.fusion_min_.transform(x_min, x_min_abs, x_min_ord)
        s_oa, s_or, s_ar = self.fusion_soo_.transform(x_soo, x_soo_abs, x_soo_ord)

        # Concatenate all 8 multiviews
        return np.hstack([x_min, x_soo, m_oa, m_or, m_ar, s_oa, s_or, s_ar])

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        return {
            "max_shapelets": 20,
            "per_shapelets": 5,
            "max_iter": 2,
            "random_state": 42,
        }
