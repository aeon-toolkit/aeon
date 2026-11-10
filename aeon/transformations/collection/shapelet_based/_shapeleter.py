"""Shapeleter transformation."""

__maintainer__ = []
__all__ = ["ShapeleterTransform"]

import numpy as np
from numba import njit, prange
from scipy import linalg
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
    # same zero handling as sklearn.preprocessing.scale
    std = np.where(std < 10 * np.finfo(np.float64).eps, 1.0, std)
    return (s - mean) / std


# Mirrors scikit-learn's LinearDiscriminantAnalysis(solver="svd", n_components=1)
# for many independent 2-feature problems; test_sk_lda_pair_matches_sklearn
# guards against a scikit-learn change.
def _sk_lda_pair_fit(x1, x2, y, tol=1e-4):
    """Fit K independent 2-feature LDA problems like sklearn's LDA(solver='svd').

    Parameters
    ----------
    x1, x2 : np.ndarray, shape (n_cases, K)
        The two features of each of the K problems.
    y : np.ndarray, shape (n_cases,)
        Class labels.

    Returns
    -------
    dict with ``ok`` (K,) bool, ``xbar1``, ``xbar2``, ``s1``, ``s2`` such that
    the projection is ``(x1 - xbar1) * s1 + (x2 - xbar2) * s2``. ``ok`` is
    False where scikit-learn would raise or return an empty projection
    (the reference then uses the mean of the two columns).
    """
    n, k = x1.shape
    classes = np.unique(y)
    n_classes = len(classes)
    ok = np.zeros(k, dtype=bool)
    s1 = np.zeros(k)
    s2 = np.zeros(k)
    if n_classes < 2 or n <= n_classes:
        zeros = np.zeros(k)
        return {"ok": ok, "xbar1": zeros, "xbar2": zeros, "s1": s1, "s2": s2}
    pri = np.array([np.mean(y == c) for c in classes])
    x = np.stack([x1, x2], axis=2).transpose(1, 0, 2)  # (K, n, 2)
    means = np.stack(
        [x[:, y == c, :].mean(axis=1) for c in classes], axis=1
    )  # (K, C, 2)
    xbar = np.einsum("c,kcf->kf", pri, means)
    xc = x.copy()
    for i, c in enumerate(classes):
        xc[:, y == c, :] -= means[:, i : i + 1, :]
    std = xc.std(axis=1)
    std[std == 0] = 1.0
    # scikit-learn concatenates the class-centred groups; the row order
    # changes the SVD sign.
    order = np.concatenate([np.where(y == c)[0] for c in classes])
    xs = np.sqrt(1.0 / n) * (xc[:, order, :] / std[:, None, :])
    wts = np.sqrt((n * pri) * (1.0 / (n_classes - 1)))
    cm = wts[None, :, None] * (means - xbar[:, None, :])  # (K, C, 2)
    for j in range(k):
        try:
            _, s, vt = linalg.svd(xs[j], full_matrices=False)
            rank = int(np.sum(s > tol))
            if rank == 0:
                continue
            sc = (vt[:rank, :] / std[j]).T / s[:rank]
            _, s_b, vt_b = linalg.svd(cm[j] @ sc, full_matrices=False)
            rank_b = int(np.sum(s_b > tol * s_b[0]))
            if rank_b < 1:
                continue
            scal = sc @ vt_b.T[:, :rank_b]
            ok[j] = True
            s1[j] = scal[0, 0]
            s2[j] = scal[1, 0]
        except (linalg.LinAlgError, ValueError):
            continue
    return {"ok": ok, "xbar1": xbar[:, 0], "xbar2": xbar[:, 1], "s1": s1, "s2": s2}


class _PairLDA:
    """One fused view: LDA projection, mean fallback, one-sided clipping."""

    def __init__(self, threshold=20.0):
        self.threshold = threshold

    def fit(self, x1, x2, y):
        self.params_ = _sk_lda_pair_fit(x1, x2, y)
        return self

    def project(self, x1, x2):
        p = self.params_
        out = 0.5 * (x1 + x2)
        ok = p["ok"]
        out[:, ok] = (x1[:, ok] - p["xbar1"][ok]) * p["s1"][ok] + (
            x2[:, ok] - p["xbar2"][ok]
        ) * p["s2"][ok]
        out[out > self.threshold] = self.threshold
        return out


class _CombinationFusion:
    """Pairwise LDA fusion of original, absolute and ordinal positional views.

    For each shapelet, pairs of the three views (original/absolute, original/ordinal,
    absolute/ordinal) are projected to one dimension by a two-feature linear
    discriminant, values above ``threshold`` are clipped (one-sided), and each fused
    block is scaled to [-1, 1] with ``MinMaxScaler``.

    Notes
    -----
    The pairwise projection reproduces scikit-learn's
    ``LinearDiscriminantAnalysis(n_components=1)`` (SVD solver) applied to each feature
    pair, including its sign convention, but fits all pairs in one pass. Pairs for which
    scikit-learn would fail use the mean of the two features, as in the reference
    implementation.
    """

    def __init__(self, threshold=20.0):
        self.threshold = threshold
        self.pair_oa = _PairLDA(threshold=threshold)
        self.pair_or = _PairLDA(threshold=threshold)
        self.pair_ar = _PairLDA(threshold=threshold)
        self.scaler_oa = MinMaxScaler((-1, 1))
        self.scaler_or = MinMaxScaler((-1, 1))
        self.scaler_ar = MinMaxScaler((-1, 1))

    def fit_transform(self, x_orig, x_abs, x_ord, y):
        """Fit LDA weights and scale transformed features to [-1, 1]."""
        self.pair_oa.fit(x_orig, x_abs, y)
        self.pair_or.fit(x_orig, x_ord, y)
        self.pair_ar.fit(x_abs, x_ord, y)

        out_oa = self.pair_oa.project(x_orig, x_abs)
        out_or = self.pair_or.project(x_orig, x_ord)
        out_ar = self.pair_ar.project(x_abs, x_ord)

        return (
            self.scaler_oa.fit_transform(out_oa),
            self.scaler_or.fit_transform(out_or),
            self.scaler_ar.fit_transform(out_ar),
        )

    def transform(self, x_orig, x_abs, x_ord):
        """Transform feature pairs with fitted LDA weights."""
        out_oa = self.pair_oa.project(x_orig, x_abs)
        out_or = self.pair_or.project(x_orig, x_ord)
        out_ar = self.pair_ar.project(x_abs, x_ord)

        return (
            self.scaler_oa.transform(out_oa),
            self.scaler_or.transform(out_or),
            self.scaler_ar.transform(out_ar),
        )


class ShapeleterTransform(BaseCollectionTransformer):
    """
    Shapeleter transformer: hypergraph shapelet pruning and dual positional embedding.

    Shapeleter [1]_ enhances shapelet-based representations. Candidate shapelets are
    sampled with a random dilated shapelet transform (RDST), pruned with a
    neighbourhood hypergraph (purity and Jaccard diversity), and each retained
    shapelet is described by its minimum distance, its occurrence count and two
    sinusoidal positional encodings (absolute position and ordinal rank). The three
    views are fused pairwise with a two-feature linear discriminant.

    Parameters
    ----------
    max_shapelets : int, default=20000
        Number of candidate shapelets sampled by the RDST backbone.
    per_shapelets : int, default=2500
        Target total number of retained shapelets. It is converted to a global keep
        rate ``per_shapelets / n_candidates`` that is applied to every
        (length, dilation, normalisation) group, rounding up, so the total can be
        slightly larger than ``per_shapelets``.
    alpha_similarity : float, default=0.0
        Similarity parameter of the RDST candidate sampling.
    proba_normalization : float, default=0.8
        Probability that a sampled shapelet is z-normalised.
    neighbor_k : tuple of int, default=(3, 5)
        Neighbourhood sizes used to build the hyperedges of the pruning hypergraph.
        Values larger than the number of candidates in a group are ignored.
    jaccard_threshold : float, default=0.9
        Two shapelets whose hyperedge sets have a Jaccard similarity at or above this
        value are treated as redundant during pruning. Must be in (0, 1].
    ka : float, default=8.0
        Initial value of the absolute-position scale ``Ka``.
    ko : float, default=8.0
        Initial value of the ordinal-position scale ``Ko``.
    k_min : float, default=1.0
        Lower bound of ``Ka`` and ``Ko``.
    k_max : float, default=15.0
        Upper bound of ``Ka`` and ``Ko``.
    lr : float, default=0.002
        Learning rate of the Adam optimiser used for ``Ka`` and ``Ko``.
    max_iter : int, default=200
        Maximum number of Adam iterations per dilation. If 0, ``ka`` and ``ko`` are
        used unchanged. The optimisation stops, keeping the last values inside
        ``[k_min, k_max]``, as soon as ``Ka`` or ``Ko`` leaves that interval.
    random_state : int, RandomState instance or None, default=None
        Seed used by the RDST backbone.

    Attributes
    ----------
    rdst_ : RandomDilatedShapeletTransform
        The fitted RDST backbone, whose shapelets have been pruned.
    series_len_ : int
        Length of the training series.
    shapelet_lengths_ : np.ndarray
        Length of every retained shapelet.
    shapelet_dilations_ : np.ndarray
        Dilation of every retained shapelet.
    unique_dilations_ : np.ndarray
        Distinct dilations of the retained shapelets.
    ka_opt_ : dict
        Optimised ``Ka`` for every dilation.
    ko_opt_ : dict
        Optimised ``Ko`` for every dilation.
    base_abs_ : dict
        Absolute-position frequency base for every dilation.
    base_ord_ : dict
        Ordinal-position frequency base for every dilation.
    scaler_min_ : MinMaxScaler
        Scaler of the minimum-distance features.
    scaler_soo_ : MinMaxScaler
        Scaler of the shapelet-occurrence features.
    fusion_min_ : _CombinationFusion
        Pairwise LDA fusion of the minimum-distance views.
    fusion_soo_ : _CombinationFusion
        Pairwise LDA fusion of the shapelet-occurrence views.

    Notes
    -----
    The output has eight blocks of one column per retained shapelet: the scaled
    minimum distance, the scaled occurrence, and for the minimum distance and the
    occurrence the original/absolute, original/ordinal and absolute/ordinal fusions.
    ``Ka`` and ``Ko`` are optimised per dilation; the absolute-position scale of a
    dilation uses the length of the first retained shapelet with that dilation (RDST
    draws a single shapelet length by default). Pruning only compares the first
    channel, so the transformer is univariate. The projection of the fusion mirrors
    scikit-learn's ``LinearDiscriminantAnalysis``, see ``_CombinationFusion``.
    The Hilbert-type series transforms used by ``ShapeleterClassifier`` are
    quadratic in the series length.

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
        if not isinstance(self.jaccard_threshold, (int, float, np.floating)) or not (
            0.0 < self.jaccard_threshold <= 1.0
        ):
            raise ValueError("jaccard_threshold must be in (0, 1].")
        if (
            not isinstance(self.neighbor_k, (tuple, list))
            or len(self.neighbor_k) == 0
            or any(
                not isinstance(k, (int, np.integer)) or k <= 0 for k in self.neighbor_k
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
                    for norm_flag in [1, 0]:
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
