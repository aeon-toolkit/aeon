"""Extended Isolation Forest (EIF) for anomaly detection."""

__maintainer__ = []
__all__ = ["ExtendedIsolationForest"]

import numpy as np
from sklearn.utils import check_random_state

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.utils.windowing import reverse_windowing, sliding_windows


def _c_factor(n: int) -> float:
    """Average path length of an unsuccessful search in a binary search tree.

    This is the normalisation constant ``c(n)`` from the Isolation Forest paper, equal
    to the expected path length of an unsuccessful search in a BST of ``n`` nodes. It is
    used to normalise the path lengths so that anomaly scores are comparable across
    sub-sample sizes.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # H(n-1) = ln(n-1) + Euler-Mascheroni constant
    harmonic = np.log(n - 1) + np.euler_gamma
    return 2.0 * harmonic - 2.0 * (n - 1) / n


class _ExtendedIsolationTree:
    """A single extended isolation tree using random hyperplane splits.

    Unlike a standard isolation tree, which splits on a single randomly chosen feature
    (an axis-parallel cut), an extended isolation tree splits on a random hyperplane
    ``(x - p) . n <= 0``. The number of non-zero components of the normal vector ``n``
    is controlled by ``extension_level``: ``extension_level=0`` zeroes all but one
    component and recovers the original axis-parallel Isolation Forest, while
    ``extension_level = n_features - 1`` uses fully oriented hyperplanes.
    """

    def __init__(self, height_limit: int, extension_level: int, rng):
        self._height_limit = height_limit
        self._extension_level = extension_level
        self._rng = rng
        self._root = None

    def fit(self, X: np.ndarray) -> "_ExtendedIsolationTree":
        self._root = self._grow(X, current_height=0)
        return self

    def _grow(self, X: np.ndarray, current_height: int) -> dict:
        n_samples, n_features = X.shape
        if current_height >= self._height_limit or n_samples <= 1:
            return {"leaf": True, "size": n_samples}

        # Random hyperplane: normal vector ``n`` and an intercept point ``p`` drawn
        # uniformly from the bounding box of the data reaching this node.
        normal = self._rng.normal(size=n_features)
        n_zeroed = n_features - self._extension_level - 1
        if n_zeroed > 0:
            zeroed = self._rng.choice(n_features, size=n_zeroed, replace=False)
            normal[zeroed] = 0.0

        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        intercept = self._rng.uniform(mins, maxs)

        projection = (X - intercept) @ normal
        left_mask = projection <= 0
        # Degenerate split (all points on one side, e.g. constant window): make a leaf
        # rather than recursing forever on an unsplittable set.
        if left_mask.all() or (~left_mask).all():
            return {"leaf": True, "size": n_samples}

        return {
            "leaf": False,
            "normal": normal,
            "intercept": intercept,
            "left": self._grow(X[left_mask], current_height + 1),
            "right": self._grow(X[~left_mask], current_height + 1),
        }

    def path_length(self, X: np.ndarray) -> np.ndarray:
        """Return the path length of each row of ``X`` through this tree."""
        lengths = np.empty(X.shape[0], dtype=float)
        self._path_length(
            X,
            self._root,
            current_length=0.0,
            indices=np.arange(X.shape[0]),
            out=lengths,
        )
        return lengths

    def _path_length(self, X, node, current_length, indices, out) -> None:
        if node["leaf"]:
            out[indices] = current_length + _c_factor(node["size"])
            return
        projection = (X[indices] - node["intercept"]) @ node["normal"]
        left = projection <= 0
        if left.any():
            self._path_length(X, node["left"], current_length + 1.0, indices[left], out)
        if (~left).any():
            self._path_length(
                X, node["right"], current_length + 1.0, indices[~left], out
            )


class ExtendedIsolationForest(BaseSeriesAnomalyDetector):
    """Extended Isolation Forest (EIF) for anomaly detection.

    The Extended Isolation Forest [1]_ generalises the Isolation Forest [2]_ by
    isolating observations with randomly oriented hyperplanes instead of axis-parallel
    cuts. Axis-parallel splitting introduces artefacts in the resulting anomaly score
    map (bands of low score aligned with the feature axes); using random hyperplanes
    removes this bias and yields a more consistent score field, while retaining the
    linear-time isolation mechanism.

    Anomalies are isolated closer to the root of each tree and therefore have shorter
    expected path lengths. The anomaly score of a point ``x`` is
    ``s(x) = 2 ** (-E[h(x)] / c(psi))``, where ``E[h(x)]`` is the mean path length of
    ``x`` across the forest, ``psi`` is the sub-sampling size and ``c`` is the expected
    path length of an unsuccessful binary-search-tree search. Scores approach ``1`` for
    anomalies and ``0.5`` or below for normal points.

    The detector operates on sliding windows of the input series (see ``window_size``
    and ``stride``). For multivariate series the channels within a window are stacked
    into a single feature vector, so each window has ``window_size * n_channels``
    features.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of isolation trees in the ensemble.
    max_samples : int, float or "auto", default="auto"
        The number of windows to draw (without replacement) to train each tree.

        - If ``int``, draw ``max_samples`` windows.
        - If ``float``, draw ``max_samples * n_windows`` windows.
        - If ``"auto"``, use ``min(256, n_windows)``.
    extension_level : int or None, default=None
        The degree of freedom of the splitting hyperplanes. Must be between ``0`` and
        ``n_features - 1``, where ``n_features = window_size * n_channels``. ``0``
        recovers the standard axis-parallel Isolation Forest; ``n_features - 1`` uses
        fully extended hyperplanes. If ``None``, the maximum value ``n_features - 1``
        is used.
    random_state : int, np.random.RandomState or None, default=None
        Seed or random state for reproducibility of the sub-sampling and the random
        hyperplanes.
    window_size : int, default=10
        Size of the sliding window.
    stride : int, default=1
        Stride of the sliding window.

    References
    ----------
    .. [1] Hariri, S., Kind, M. C., & Brunner, R. J. (2021). Extended Isolation Forest.
       IEEE Transactions on Knowledge and Data Engineering, 33(4), 1479-1489.
       https://doi.org/10.1109/TKDE.2019.2947676
    .. [2] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. In 2008
       Eighth IEEE International Conference on Data Mining (pp. 413-422).

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.outlier_detection import (
    ...     ExtendedIsolationForest,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=200)
    >>> X[100] = 10.0
    >>> detector = ExtendedIsolationForest(window_size=1, random_state=0)
    >>> scores = detector.fit_predict(X)
    >>> int(np.argmax(scores))
    100
    """

    _tags = {
        "capability:multivariate": True,
        "capability:univariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
        "learning_type:semi_supervised": True,
    }

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples="auto",
        extension_level: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        window_size: int = 10,
        stride: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state
        self.window_size = window_size
        self.stride = stride

        super().__init__(axis=0)

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        _X, _ = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        self._inner_fit(_X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        window_scores = self._score(_X)
        point_scores = reverse_windowing(
            window_scores, self.window_size, np.nanmean, self.stride, padding
        )
        return point_scores

    def _fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        _X, padding = sliding_windows(
            X, window_size=self.window_size, stride=self.stride, axis=0
        )
        self._inner_fit(_X)
        window_scores = self._score(_X)
        point_scores = reverse_windowing(
            window_scores, self.window_size, np.nanmean, self.stride, padding
        )
        return point_scores

    def _inner_fit(self, X: np.ndarray) -> None:
        rng = check_random_state(self.random_state)
        n_windows, n_features = X.shape

        if isinstance(self.max_samples, str):
            if self.max_samples != "auto":
                raise ValueError(
                    "max_samples must be an int, a float or 'auto', got "
                    f"'{self.max_samples}'"
                )
            subsample_size = min(256, n_windows)
        elif isinstance(self.max_samples, float):
            subsample_size = int(self.max_samples * n_windows)
        else:
            subsample_size = int(self.max_samples)
        subsample_size = max(1, min(subsample_size, n_windows))

        if self.extension_level is None:
            extension_level = n_features - 1
        else:
            extension_level = self.extension_level
        if not 0 <= extension_level <= n_features - 1:
            raise ValueError(
                "extension_level must be between 0 and n_features - 1 "
                f"({n_features - 1}), got {self.extension_level}"
            )

        height_limit = max(1, int(np.ceil(np.log2(max(subsample_size, 2)))))

        self._subsample_size = subsample_size
        self.trees_ = []
        for _ in range(self.n_estimators):
            if subsample_size < n_windows:
                idx = rng.choice(n_windows, size=subsample_size, replace=False)
                sample = X[idx]
            else:
                sample = X
            tree = _ExtendedIsolationTree(height_limit, extension_level, rng)
            self.trees_.append(tree.fit(sample))

    def _score(self, X: np.ndarray) -> np.ndarray:
        mean_path = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees_:
            mean_path += tree.path_length(X)
        mean_path /= len(self.trees_)
        return 2.0 ** (-mean_path / _c_factor(self._subsample_size))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict
            Parameters to create testing instances of the class.
        """
        return {"n_estimators": 5, "window_size": 10}
