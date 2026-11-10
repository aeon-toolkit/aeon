"""Shapeleter transformation."""

__maintainer__ = []
__all__ = ["ShapeleterTransform"]

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.shapelet_based._dilated_shapelet_transform import (
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


def _knn_hyperedges(cleaned_sub, valid_k):
    """Compute self-inclusive k-NN hyperedges for candidate shapelets."""
    n_shapelets = cleaned_sub.shape[0]
    dist_mat = euclidean_distances(cleaned_sub)
    np.fill_diagonal(dist_mat, -np.inf)
    edges = []
    for k in valid_k:
        for i in range(n_shapelets):
            edges.append(np.argsort(dist_mat[i])[:k])
    return edges


def _hypergraph_prune(
    shapelets,
    shapelet_len,
    labels,
    neighbor_k=(3, 5),
    save_rate=0.125,
    jaccard_threshold=0.9,
):
    """Prune candidate shapelets using hypergraph purity and Jaccard diversity."""
    n_shapelets = len(labels)
    valid_k = [k for k in neighbor_k if k <= n_shapelets]
    if len(valid_k) == 0 or n_shapelets <= 1:
        return np.arange(n_shapelets)

    raw_sub = shapelets[:, 0, :shapelet_len]
    cleaned_sub = np.nan_to_num(raw_sub, nan=0.0, posinf=0.0, neginf=0.0)

    edges = _knn_hyperedges(cleaned_sub, valid_k)

    purity = np.zeros(n_shapelets, dtype=float)
    degree = np.zeros(n_shapelets, dtype=float)
    for edge in edges:
        sub_labels = labels[edge]
        edge_sz = len(edge)
        for idx in edge:
            degree[idx] += 1.0
            purity[idx] += np.sum(labels[idx] == sub_labels) / float(edge_sz)

    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where(degree > 0, purity / degree, 0.0)

    j_matrix = _compute_jaccard_matrix(edges, n_shapelets)
    j_matrix_bin = np.zeros_like(j_matrix)
    j_matrix_bin[j_matrix >= jaccard_threshold] = 1.0
    np.fill_diagonal(j_matrix_bin, -1.0)

    save_n = max(1, min(int(np.ceil(save_rate * n_shapelets)), n_shapelets))

    # Follows neighbor_hypergraph_based_shapelet_selection
    # (reference notebook cell 11, CCHe64/Shapeleter) line by line.
    # Note: the reference does not remap the label index after removing
    # an exhausted label; this is kept for parity.
    unique_labels = np.unique(labels)
    top_node = int(np.argmax(purity))
    saved_nodes = [top_node]

    label_index = int(np.where(unique_labels == labels[top_node])[0][0]) + 1
    if label_index >= len(unique_labels):
        label_index = 0
    candidate_nodes = np.array(
        [node for node in np.argsort(purity)[::-1] if node != top_node]
    )

    for _ in range(save_n - 1):
        last_saved = saved_nodes[-1]
        similar_nodes = np.where(j_matrix_bin[last_saved, :] == 1.0)[0]
        j_matrix_bin[:, similar_nodes] = -1.0
        j_matrix_bin[similar_nodes, :] = -1.0
        j_matrix_bin[last_saved, :] = -1.0
        j_matrix_bin[:, last_saved] = -1.0
        candidate_nodes = candidate_nodes[~np.isin(candidate_nodes, similar_nodes)]
        if len(candidate_nodes) == 0:
            break

        cand_labels = labels[candidate_nodes]
        curr_label = unique_labels[label_index]
        if np.sum(cand_labels == curr_label) == 0:
            curr_label_0 = curr_label
            label_index += 1
            if label_index >= len(unique_labels):
                label_index = 0
            curr_label = unique_labels[label_index]
            unique_labels = unique_labels[unique_labels != curr_label_0]

        sub_cands = candidate_nodes[cand_labels == curr_label]
        if len(sub_cands) == 0:
            break
        best_cand = int(sub_cands[np.argmax(purity[sub_cands])])

        label_index += 1
        if label_index >= len(unique_labels):
            label_index = 0

        saved_nodes.append(best_cand)
        candidate_nodes = candidate_nodes[candidate_nodes != best_cand]

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
    """Optimize Ka and Ko using Adam in pure NumPy with bounds checking."""
    w = np.array([np.log(ka0), np.log(ko0)], dtype=np.float64)
    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    h = 1e-5

    last_valid_ka = float(ka0)
    last_valid_ko = float(ko0)

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
        w_next = w - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        ka_next = float(np.exp(w_next[0]))
        ko_next = float(np.exp(w_next[1]))

        if not (k_min <= ka_next <= k_max and k_min <= ko_next <= k_max):
            break

        w = w_next
        last_valid_ka = ka_next
        last_valid_ko = ko_next

    return last_valid_ka, last_valid_ko


@njit(parallel=True, fastmath=True, cache=True)
def _hilbert_transform_3d(X):
    """Compute exact discrete Hilbert transform via direct sum."""
    n_cases, n_channels, n_timepoints = X.shape
    out = np.zeros((n_cases, n_channels, n_timepoints), dtype=np.float64)
    for c in prange(n_cases):
        for ch in range(n_channels):
            for k in range(n_timepoints):
                val = 0.0
                for j in range(n_timepoints):
                    if j != k:
                        val += X[c, ch, j] / (k - j)
                out[c, ch, k] = val
    return out


def _hilbert_transform(signal):
    """Compute exact direct-sum discrete Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array with time along the last axis.

    Returns
    -------
    ht : np.ndarray
        Direct-sum discrete Hilbert transformed signal.
    """
    orig_ndim = signal.ndim
    if orig_ndim == 1:
        X = signal.reshape(1, 1, -1)
    elif orig_ndim == 2:
        X = signal.reshape(signal.shape[0], 1, -1)
    elif orig_ndim == 3:
        X = signal
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D signal, got {orig_ndim}D.")

    out = _hilbert_transform_3d(np.ascontiguousarray(X, dtype=np.float64))

    if orig_ndim == 1:
        return out[0, 0]
    elif orig_ndim == 2:
        return out[:, 0]
    return out


def _series_transform(X, mode):
    """Transform time series collection into alternative representation.

    Parameters
    ----------
    X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
        Input time series collection.
    mode : {"F", "H", "HF"}
        Transformation mode:
        - "F": First-order difference series (T0).
        - "H": Exact discrete Hilbert transform (T1).
        - "HF": Difference of Hilbert transform (T2 = diff(HT)).

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
        s = np.diff(_hilbert_transform(X), axis=-1)
    else:
        raise ValueError(
            f"Unknown transformation mode: {mode}. Expected 'F', 'H', or 'HF'."
        )

    mean = np.mean(s, axis=-1, keepdims=True)
    std = np.std(s, axis=-1, keepdims=True)
    return (s - mean) / (std + 1e-8)


def _vectorized_2feature_lda_fit(X1, X2, y):
    """Compute vectorized closed-form Fisher 2-feature LDA projection weights.

    Projects each feature pair (X1[:, i], X2[:, i]) onto a 1D discriminant subspace.
    If a feature pair has near-zero variance or is collinear, falls back to the
    unweighted average.
    """
    N, K = X1.shape
    classes = np.unique(y)
    n_classes = len(classes)

    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)

    if n_classes <= 1 or N <= 2:
        return (
            np.zeros(K, dtype=bool),
            np.zeros(K),
            np.zeros(K),
            mean1,
            mean2,
            np.zeros(K),
        )

    priors = np.array([np.mean(y == c) for c in classes])
    means1 = np.zeros((n_classes, K))
    means2 = np.zeros((n_classes, K))
    sw11 = np.zeros(K)
    sw22 = np.zeros(K)
    sw12 = np.zeros(K)

    for idx, c in enumerate(classes):
        mask = y == c
        n_c = np.sum(mask)
        if n_c == 0:
            continue
        m1 = np.mean(X1[mask], axis=0)
        m2 = np.mean(X2[mask], axis=0)
        means1[idx] = m1
        means2[idx] = m2
        diff1 = X1[mask] - m1
        diff2 = X2[mask] - m2
        sw11 += np.sum(diff1**2, axis=0) / N
        sw22 += np.sum(diff2**2, axis=0) / N
        sw12 += np.sum(diff1 * diff2, axis=0) / N

    det = sw11 * sw22 - sw12**2
    valid = (det > 1e-12) & (sw11 > 1e-12) & (sw22 > 1e-12)

    w1 = np.zeros(K)
    w2 = np.zeros(K)

    if n_classes == 2:
        d_mu1 = means1[1] - means1[0]
        d_mu2 = means2[1] - means2[0]
        w1[valid] = sw22[valid] * d_mu1[valid] - sw12[valid] * d_mu2[valid]
        w2[valid] = -sw12[valid] * d_mu1[valid] + sw11[valid] * d_mu2[valid]
    else:
        overall_m1 = np.mean(X1, axis=0)
        overall_m2 = np.mean(X2, axis=0)
        sb11 = np.zeros(K)
        sb22 = np.zeros(K)
        sb12 = np.zeros(K)
        for idx in range(n_classes):
            dm1 = means1[idx] - overall_m1
            dm2 = means2[idx] - overall_m2
            p = priors[idx]
            sb11 += p * (dm1**2)
            sb22 += p * (dm2**2)
            sb12 += p * (dm1 * dm2)

        inv_det = np.zeros(K)
        inv_det[valid] = 1.0 / det[valid]

        m11 = (sw22 * sb11 - sw12 * sb12) * inv_det
        m12 = (sw22 * sb12 - sw12 * sb22) * inv_det
        m21 = (-sw12 * sb11 + sw11 * sb12) * inv_det
        m22 = (-sw12 * sb12 + sw11 * sb22) * inv_det

        tr = m11 + m22
        disc = np.maximum(0.0, (m11 - m22) ** 2 + 4.0 * m12 * m21)
        lam = 0.5 * (tr + np.sqrt(disc))

        mask_m12 = valid & (np.abs(m12) > 1e-12)
        w1[mask_m12] = m12[mask_m12]
        w2[mask_m12] = lam[mask_m12] - m11[mask_m12]

        mask_m21 = valid & (~mask_m12) & (np.abs(m21) > 1e-12)
        w1[mask_m21] = lam[mask_m21] - m22[mask_m21]
        w2[mask_m21] = m21[mask_m21]

        mask_diag = valid & (~mask_m12) & (~mask_m21)
        w1[mask_diag & (m11 >= m22)] = 1.0
        w2[mask_diag & (m11 < m22)] = 1.0

    norm_w = np.sqrt(w1**2 + w2**2)
    norm_valid = valid & (norm_w > 1e-12)
    w1[norm_valid] /= norm_w[norm_valid]
    w2[norm_valid] /= norm_w[norm_valid]

    if n_classes > 2:
        # For multiclass, match positive w1 convention since SVD sign varies
        flip = norm_valid & ((w1 < 0) | ((np.abs(w1) <= 1e-12) & (w2 < 0)))
        w1[flip] = -w1[flip]
        w2[flip] = -w2[flip]
    # For n_classes == 2, do NOT apply the w1 >= 0 flip: the binary formula
    # already gives w . (mu1 - mu0) > 0, matching sklearn's convention.

    scale = np.zeros(K)
    quad = (
        w1**2 * sw11
        + 2.0 * w1 * w2 * sw12
        + w2**2 * sw22
    )
    scale_valid = norm_valid & (quad > 1e-12)
    scale[scale_valid] = 1.0 / np.sqrt(quad[scale_valid])

    valid = scale_valid
    return valid, w1, w2, mean1, mean2, scale


def _vectorized_2feature_lda_transform(X1, X2, valid, w1, w2, mean1, mean2, scale):
    """Project feature pairs with fitted LDA weights or fallback to average."""
    out = np.zeros_like(X1)
    out[:, valid] = (
        (X1[:, valid] - mean1[valid]) * w1[valid]
        + (X2[:, valid] - mean2[valid]) * w2[valid]
    ) * scale[valid]
    out[:, ~valid] = 0.5 * (X1[:, ~valid] + X2[:, ~valid])
    return out


class _CombinationFusion:
    """Pairwise 2-feature LDA fusion of original, absolute, and ordinal views.

    Notes
    -----
    The closed-form projection reproduces scikit-learn's
    `LinearDiscriminantAnalysis(n_components=1).transform` (centred, unit
    within-class variance); for more than 2 classes the overall sign is
    fixed by convention.
    """

    def __init__(self, threshold=20.0):
        self.threshold = threshold
        self.scaler_oa = MinMaxScaler((-1, 1))
        self.scaler_or = MinMaxScaler((-1, 1))
        self.scaler_ar = MinMaxScaler((-1, 1))

    def fit_transform(self, x_orig, x_abs, x_ord, y):
        """Fit LDA weights and scale transformed features to [-1, 1]."""
        (
            self.valid_oa,
            self.w1_oa,
            self.w2_oa,
            self.mean1_oa,
            self.mean2_oa,
            self.scale_oa,
        ) = _vectorized_2feature_lda_fit(x_orig, x_abs, y)
        (
            self.valid_or,
            self.w1_or,
            self.w2_or,
            self.mean1_or,
            self.mean2_or,
            self.scale_or,
        ) = _vectorized_2feature_lda_fit(x_orig, x_ord, y)
        (
            self.valid_ar,
            self.w1_ar,
            self.w2_ar,
            self.mean1_ar,
            self.mean2_ar,
            self.scale_ar,
        ) = _vectorized_2feature_lda_fit(x_abs, x_ord, y)

        out_oa = _vectorized_2feature_lda_transform(
            x_orig,
            x_abs,
            self.valid_oa,
            self.w1_oa,
            self.w2_oa,
            self.mean1_oa,
            self.mean2_oa,
            self.scale_oa,
        )
        out_or = _vectorized_2feature_lda_transform(
            x_orig,
            x_ord,
            self.valid_or,
            self.w1_or,
            self.w2_or,
            self.mean1_or,
            self.mean2_or,
            self.scale_or,
        )
        out_ar = _vectorized_2feature_lda_transform(
            x_abs,
            x_ord,
            self.valid_ar,
            self.w1_ar,
            self.w2_ar,
            self.mean1_ar,
            self.mean2_ar,
            self.scale_ar,
        )

        out_oa[out_oa > self.threshold] = self.threshold
        out_or[out_or > self.threshold] = self.threshold
        out_ar[out_ar > self.threshold] = self.threshold

        return (
            self.scaler_oa.fit_transform(out_oa),
            self.scaler_or.fit_transform(out_or),
            self.scaler_ar.fit_transform(out_ar),
        )

    def transform(self, x_orig, x_abs, x_ord):
        """Transform feature pairs with fitted LDA weights."""
        out_oa = _vectorized_2feature_lda_transform(
            x_orig,
            x_abs,
            self.valid_oa,
            self.w1_oa,
            self.w2_oa,
            self.mean1_oa,
            self.mean2_oa,
            self.scale_oa,
        )
        out_or = _vectorized_2feature_lda_transform(
            x_orig,
            x_ord,
            self.valid_or,
            self.w1_or,
            self.w2_or,
            self.mean1_or,
            self.mean2_or,
            self.scale_or,
        )
        out_ar = _vectorized_2feature_lda_transform(
            x_abs,
            x_ord,
            self.valid_ar,
            self.w1_ar,
            self.w2_ar,
            self.mean1_ar,
            self.mean2_ar,
            self.scale_ar,
        )

        out_oa[out_oa > self.threshold] = self.threshold
        out_or[out_or > self.threshold] = self.threshold
        out_ar[out_ar > self.threshold] = self.threshold

        return (
            self.scaler_oa.transform(out_oa),
            self.scaler_or.transform(out_or),
            self.scaler_ar.transform(out_ar),
        )


class ShapeleterTransform(BaseCollectionTransformer):
    """Shapeleter transformer with hypergraph pruning and dual positional embedding.

    Shapeleter [1]_ enhances shapelet-based time series representations by combining:
    1. A random dilated shapelet transform (RDST) candidate generation backbone.
    2. Multi-scale hypergraph pruning grouped by shapelet length, dilation, and
       normalization, utilizing self-inclusive k-NN hyperedges, hypergraph purity
       weighting, and Jaccard-based diversity filtering.
    3. Dual sinusoidal positional embeddings incorporating both absolute occurrence
       positions (SAPE) and ordinal relative ranks (SOPE), with frequency scaling
       parameters optimized via log-space Adam.
    4. Multi-view feature fusion via pairwise 2-feature Linear Discriminant Analysis
       (LDA) with one-sided threshold clipping and MinMax normalization.

    Parameters
    ----------
    max_shapelets : int, default=20000
        Maximum number of candidate shapelets sampled initially by the RDST backbone.
    per_shapelets : int, default=2500
        Target number of shapelets preserved per group after hypergraph pruning.
    alpha_similarity : float, default=0.0
        The minimum similarity between shapelets for self-similarity pruning in RDST.
    proba_normalization : float, default=0.8
        Probability of normalizing shapelets during extraction in RDST.
    neighbor_k : tuple of int, default=(3, 5)
        Neighborhood sizes for constructing hyperedges in hypergraph candidate pruning.
    jaccard_threshold : float, default=0.9
        Maximum Jaccard similarity threshold for candidate diversity filtering.
    ka : float, default=8.0
        Initial absolute positional scaling parameter (:math:`K_{a0}`).
    ko : float, default=8.0
        Initial ordinal positional scaling parameter (:math:`K_{o0}` / :math:`K_{r0}`).
    k_min : float, default=1.0
        Lower bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    k_max : float, default=15.0
        Upper bound for :math:`K_a` and :math:`K_o` during Adam optimization.
    lr : float, default=0.002
        Learning rate for Adam optimization of :math:`K_a` and :math:`K_o`.
    max_iter : int, default=200
        Maximum number of iterations for Adam positional optimization per dilation.
        If set to 0, optimization is skipped and fixed `ka` and `ko` values are used.
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

    Notes
    -----
    `per_shapelets` is converted to a global keep-rate (`per_shapelets / n_candidates`)
    applied per candidate group with `ceil`, so the total number of retained shapelets
    can slightly exceed `per_shapelets`.

    Ka and Ko are optimised per dilation. The absolute-position scale for a
    dilation uses the length of the first retained shapelet with that dilation.
    Pruning is grouped by (length, dilation, normalisation), whereas the reference
    groups by (dilation, normalisation); these are identical whenever a single
    shapelet length is used.

    References
    ----------
    .. [1] Chang-Cheng He et al. "Shapeleter: Shapelet Enhancer via Multiview
       Positional Embedding for Time Series Classification", Knowledge-Based Systems,
       2026.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.shapelet_based import ShapeleterTransform
    >>> X = np.random.RandomState(42).normal(size=(10, 1, 50))
    >>> y = np.array([0, 1] * 5)
    >>> st = ShapeleterTransform(
    ...     max_shapelets=50, per_shapelets=5, max_iter=2, random_state=42
    ... )
    >>> Xt = st.fit_transform(X, y)
    >>> Xt.shape[0] == 10
    True
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "X_inner_type": "numpy3D",
        "python_dependencies": None,
        "algorithm_type": "shapelet",
        "requires_y": True,
    }

    def __init__(
        self,
        max_shapelets=20000,
        per_shapelets=2500,
        alpha_similarity=0.0,
        proba_normalization=0.8,
        neighbor_k=(3, 5),
        jaccard_threshold=0.9,
        ka=8.0,
        ko=8.0,
        k_min=1.0,
        k_max=15.0,
        lr=0.002,
        max_iter=200,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.per_shapelets = per_shapelets
        self.alpha_similarity = alpha_similarity
        self.proba_normalization = proba_normalization
        self.neighbor_k = neighbor_k
        self.jaccard_threshold = jaccard_threshold
        self.ka = ka
        self.ko = ko
        self.k_min = k_min
        self.k_max = k_max
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__()

    def _fit_groups(self, X, y):
        """Fit RDST backbone, pruning, positions, and fusion models; return 8 groups."""
        if y is None:
            raise ValueError(
                "ShapeleterTransform requires y for hypergraph pruning and LDA fusion."
            )

        if self.max_shapelets <= 0:
            raise ValueError("max_shapelets must be positive.")
        if self.per_shapelets <= 0:
            raise ValueError("per_shapelets must be positive.")
        if (
            not isinstance(self.jaccard_threshold, (int, float, np.floating))
            or not (0.0 < self.jaccard_threshold <= 1.0)
        ):
            raise ValueError("jaccard_threshold must be in (0, 1].")
        if (
            not isinstance(self.neighbor_k, (tuple, list))
            or len(self.neighbor_k) == 0
            or any(
                not isinstance(k, (int, np.integer)) or k <= 0
                for k in self.neighbor_k
            )
        ):
            raise ValueError("neighbor_k must contain positive integers.")
        if self.k_min <= 0:
            raise ValueError("k_min must be positive.")
        if self.k_min > self.k_max:
            raise ValueError("k_min must be <= k_max.")
        if not (self.k_min <= self.ka <= self.k_max):
            raise ValueError("ka must be within [k_min, k_max].")
        if not (self.k_min <= self.ko <= self.k_max):
            raise ValueError("ko must be within [k_min, k_max].")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")

        self.series_len_ = X.shape[-1]
        self.rdst_ = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            alpha_similarity=self.alpha_similarity,
            proba_normalization=self.proba_normalization,
            random_state=self.random_state,
        )
        self.rdst_.fit(X, y)

        # RDST ``shapelets_`` tuple layout
        # (verified against _dilated_shapelet_transform.py):
        # 0: shapelet values, 1: startpoints, 2: lengths, 3: dilations,
        # 4: thresholds, 5: normalisation flags, 6: means, 7: stds, 8: source class
        shp_vals_idx = 0
        shp_lens_idx = 2
        shp_dils_idx = 3
        shp_norm_idx = 5
        shp_cls_idx = 8

        raw_shapelets = self.rdst_.shapelets_[shp_vals_idx]
        n_candidates = len(raw_shapelets)

        if n_candidates > 0 and len(np.unique(y)) > 1:
            save_rate = float(self.per_shapelets) / max(1.0, float(n_candidates))
            lengths = self.rdst_.shapelets_[shp_lens_idx]
            dilas = self.rdst_.shapelets_[shp_dils_idx]
            norms = self.rdst_.shapelets_[shp_norm_idx]
            shapelet_classes = self.rdst_.shapelets_[shp_cls_idx]
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
                                neighbor_k=self.neighbor_k,
                                save_rate=save_rate,
                                jaccard_threshold=self.jaccard_threshold,
                            )
                            selected_idx.append(sub_idx[pruned_rel])

            if len(selected_idx) > 0:
                selected_idx = np.hstack(selected_idx)
                self.rdst_.shapelets_ = tuple(
                    attr[selected_idx] for attr in self.rdst_.shapelets_
                )

        self.shapelet_lengths_ = self.rdst_.shapelets_[shp_lens_idx]
        self.shapelet_dilations_ = self.rdst_.shapelets_[shp_dils_idx]

        raw_feats = self.rdst_.transform(X)
        raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)
        raw_feats = self._normalise_soo(raw_feats)

        x_min = raw_feats[:, 0::3]
        x_arg = raw_feats[:, 1::3]
        x_soo = raw_feats[:, 2::3]

        self.scaler_min_ = MinMaxScaler((-1, 1)).fit(x_min)
        self.scaler_soo_ = MinMaxScaler((-1, 1)).fit(x_soo)
        x_min_scaled = self.scaler_min_.transform(x_min)
        x_soo_scaled = self.scaler_soo_.transform(x_soo)

        self.unique_dilations_ = np.unique(self.shapelet_dilations_)
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

        x_min_abs, x_min_ord, x_soo_abs, x_soo_ord = self._encode_positions(
            x_min_scaled, x_soo_scaled, x_arg
        )

        self.fusion_min_ = _CombinationFusion()
        self.fusion_soo_ = _CombinationFusion()

        m_oa, m_or, m_ar = self.fusion_min_.fit_transform(
            x_min_scaled, x_min_abs, x_min_ord, y
        )
        s_oa, s_or, s_ar = self.fusion_soo_.fit_transform(
            x_soo_scaled, x_soo_abs, x_soo_ord, y
        )

        return (
            x_min_scaled,
            x_soo_scaled,
            m_oa,
            s_oa,
            m_or,
            s_or,
            m_ar,
            s_ar,
        )

    def _fit(self, X, y=None):
        """Fit RDST backbone, apply hypergraph pruning, and initialize fusions."""
        self._fit_groups(X, y)
        return self

    def _fit_transform_groups(self, X, y=None):
        """Fit the transformer and return multi-view feature groups."""
        return self._fit_groups(X, y)

    def _fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        groups = self._fit_transform_groups(X, y)
        return np.hstack(groups)

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

            pos_abs = x_arg[:, sub_idx]
            x_min_abs[:, sub_idx] += np.sin(pos_abs / base_abs)
            x_soo_abs[:, sub_idx] += np.cos(pos_abs / base_abs)

            pos_ord = _assign_order_batch(pos_abs)

            x_min_ord[:, sub_idx] += np.sin(pos_ord / base_ord)
            x_soo_ord[:, sub_idx] += np.cos(pos_ord / base_ord)

        return x_min_abs, x_min_ord, x_soo_abs, x_soo_ord

    def _normalise_soo(self, raw_feats):
        """Divide each SOO column by its number of valid positions."""
        if len(self.shapelet_lengths_) > 0:
            denom = (
                self.series_len_
                - (self.shapelet_lengths_ - 1) * self.shapelet_dilations_
            )
            denom = np.maximum(1.0, denom.astype(float))
            raw_feats[:, 2::3] /= denom[np.newaxis, :]
        return raw_feats

    def _transform_groups(self, X):
        """Transform time series into 8 distinct shapelet feature groups."""
        raw_feats = self.rdst_.transform(X)
        raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)
        raw_feats = self._normalise_soo(raw_feats)

        x_min = self.scaler_min_.transform(raw_feats[:, 0::3])
        x_soo = self.scaler_soo_.transform(raw_feats[:, 2::3])
        x_arg = raw_feats[:, 1::3]

        x_min_abs, x_min_ord, x_soo_abs, x_soo_ord = self._encode_positions(
            x_min, x_soo, x_arg
        )

        m_oa, m_or, m_ar = self.fusion_min_.transform(x_min, x_min_abs, x_min_ord)
        s_oa, s_or, s_ar = self.fusion_soo_.transform(x_soo, x_soo_abs, x_soo_ord)

        return x_min, x_soo, m_oa, s_oa, m_or, s_or, m_ar, s_ar

    def _transform(self, X, y=None):
        """Transform time series into concatenated multi-view shapelet features."""
        groups = self._transform_groups(X)
        return np.hstack(groups)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        return {
            "max_shapelets": 20,
            "per_shapelets": 5,
            "max_iter": 2,
            "random_state": 42,
        }
