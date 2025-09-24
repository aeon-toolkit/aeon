"""Proximity Tree Time Series Classifier.

A decision tree classifier where the splits based on the
similarity of instances to chosen time series exemplars, measured using
aeon distances.
"""

__maintainer__ = []
__all__ = ["ProximityTree"]


import numpy as np
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.distances import distance
from aeon.utils.numba.general import (
    slope_derivative_2d,
    slope_derivative_3d,
    unique_count,
)
from aeon.utils.numba.stats import gini_gain, std


class _ProximityNode:
    """Proximity Tree node.

    Parameters
    ----------
    is_leaf: bool
        To identify leaf nodes.
    class_distribution: dict or None, default=None
        The class distribution for the node. Empty if not a leaf node.
    splitter: tuple or None, default=None
        The splitter used to split the node. Contains exemplars used, distance name and
        distance parameters. Empty if leaf node.

    Attributes
    ----------
    children: dict
        Contains the class label as a key and child node for that class. Empty for leaf
        nodes.
    """

    def __init__(
        self,
        is_leaf,
        class_distribution=None,
        splitter=None,
    ):
        self.is_leaf = is_leaf
        self.class_distribution = class_distribution
        self.splitter = splitter
        self.children = {}

        if is_leaf:
            assert class_distribution is not None
        else:
            assert splitter is not None


class ProximityTree(BaseClassifier):
    """Proximity Tree classifier.

    A Proximity Tree is a decision tree classifier where the splits based on the
    similarity of instances to chosen time series exemplars. This tree is built
    recursively, starting from the root and progressing down to the leaf nodes.

    At each internal node, a pool of candidate splitters is evaluated. Each splitter
    consists of a set of exemplar time series for each class and a parameterized
    similarity measure, both chosen randomly. The optimal splitter is selected based on
    its ability to maximize the reduction in Gini impurity, measured as the difference
    between the Gini impurity of the parent node and the weighted sum of the Gini
    impurity of the child nodes.

    Proximity Trees are particularly useful as they are the building blocks of Proximity
    Forest, the state-of-the art distance-based classifier.

    Parameters
    ----------
    n_splitters: int, default = 5
        The number of candidate splitters to be evaluated at each node.
    max_depth: int, default = None
        The maximum depth of the tree. If ``None``, then nodes are expanded until all
        leaves are pure or until all leaves contain less than ``min_samples_split``
        samples.
    min_samples_split: int, default = 2
        The minimum number of samples required to split an internal node.
    random_state : int, RandomState instance or None, default=None
        If ``int``, ``random_state`` is the seed used by the random number generator;
        If ``RandomState`` instance, ``random_state`` is the random number generator;
        If ``None``, the random number generator is the ``RandomState`` instance used
        by ``np.random``.

    Notes
    -----
    For the Java version, see
    `ProximityTree
    <https://github.com/fpetitjean/ProximityForest/blob/master/src/trees/ProximityTree.java>`_.

    References
    ----------
    .. [1] Lucas, B., Shifaz, A., Pelletier, C., O’Neill, L., Zaidi, N., Goethals, B.,
    Petitjean, F. and Webb, G.I., 2019. Proximity forest: an effective and scalable
    distance-based classifier for time series. Data Mining and Knowledge Discovery,
    33(3), pp.607-635.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import ProximityTree
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = ProximityTree(n_splitters=3)
    >>> classifier.fit(X_train, y_train)
    ProximityTree(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:unequal_length": True,
        "capability:multivariate": True,
        "algorithm_type": "distance",
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        n_splitters: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        # Reuse the transform for ddtw and wddtw
        der_X = self._get_derivatives(X)
        rng = check_random_state(self.random_state)
        self._root = self._build_tree(X, der_X, y, rng, 0)

    def _predict(self, X):
        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in self._predict_proba(X)]
        )

    def _predict_proba(self, X):
        # Reuse the transform for ddtw and wddtw
        der_X = self._get_derivatives(X)
        probas = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            probas[i] = self._traverse_tree(self._root, X[i], der_X[i])
        return probas

    def _build_tree(self, X, der_X, y, rng, depth):
        """Build the tree recursively from the root node down to the leaf nodes.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        der_X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The slope derivative of the training input samples, used for ddtw and wddtw.
        y : np.ndarray shape (n_cases,)
            The class labels for the training input samples.
        rng : np.random.RandomState
            Random number generator.
        depth : int
            The current depth of the tree, used to limit the maximum depth of the tree.

        Returns
        -------
        node : _ProximityNode
            The current node of the tree, which contains the splitter and children
            nodes.
            The root node will be returned to _fit.
        """
        # Target value in current node
        unique_classes, class_counts = unique_count(y)
        class_distribution = np.zeros(len(self.classes_))
        for i, label in enumerate(unique_classes):
            class_distribution[self._class_dictionary[label]] = class_counts[i] / len(X)

        if (
            # Pure node
            len(unique_classes) == 1
            # If min sample splits is reached
            or self.min_samples_split >= len(X)
            # If max depth is reached
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            leaf = _ProximityNode(
                is_leaf=True,
                class_distribution=class_distribution,
            )
            return leaf

        # Find the best splitter
        splitter, node_splits = self._get_best_splitter(
            X, der_X, y, unique_classes, rng
        )

        # Create root node
        node = _ProximityNode(is_leaf=False, splitter=splitter)

        # For each exemplar, create a branch
        for i, label in enumerate(unique_classes):
            child_node = self._build_tree(
                (
                    X[node_splits[i]]
                    if isinstance(X, np.ndarray)
                    else [X[j] for j in node_splits[i]]
                ),
                (
                    der_X[node_splits[i]]
                    if isinstance(der_X, np.ndarray)
                    else [der_X[j] for j in node_splits[i]]
                ),
                y[node_splits[i]],
                rng,
                depth + 1,
            )
            node.children[label] = child_node

        return node

    def _get_best_splitter(self, X, der_X, y, unique_classes, rng):
        """Get the best splitter for the current node which maximizes the gini gain.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        der_X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The slope derivative of the training input samples, used for ddtw and wddtw.
        y : np.ndarray shape (n_cases,)
            The class labels for the training input samples.
        unique_classes : np.ndarray
            The unique class labels in the training set.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        best_splitter : tuple
            The best splitter found, containing exemplars, distance name and distance
            parameters.
        best_split : list of list
            The best split found, containing the indices of the time series in each
            class.
        """
        max_gain = -np.inf
        best_splitter = None
        best_split = None
        X_std = None
        cls_idx = {}
        for label in unique_classes:
            cls_idx[label] = np.where(y == label)[0]

        for _ in range(self.n_splitters):
            exemplars, dist_name, dist_params, X_std = self._get_candidate_splitter(
                X, der_X, rng, cls_idx, X_std
            )
            splits = [[] for _ in unique_classes]

            # Use the slope derivative of the time series for ddtw and wddtw
            # use original distance to avoid recalculating every distance call
            X_used = X
            dist_used = dist_name
            if dist_name == "ddtw":
                X_used = der_X
                dist_used = "dtw"
            elif dist_name == "wddtw":
                X_used = der_X
                dist_used = "wdtw"

            # For each time series in the dataset, find the closest exemplar
            for j in range(len(X_used)):
                min_dist = np.inf
                best_exemplar_idx = None
                for i, label in enumerate(unique_classes):
                    dist = distance(
                        X_used[j],
                        exemplars[label],
                        method=dist_used,
                        **dist_params,
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_exemplar_idx = i
                splits[best_exemplar_idx].append(j)

            # Find the gini gain for this splitter separating series by closest exemplar
            # class
            y_subs = [
                np.array([y[n] for n in split], dtype=y.dtype) for split in splits
            ]
            gini_index = gini_gain(y, y_subs)
            if gini_index > max_gain:
                max_gain = gini_index
                best_splitter = (exemplars, dist_name, dist_params)
                best_split = splits

        return best_splitter, best_split

    def _traverse_tree(self, node, x, der_x):
        """Traverse the tree to find the class distribution for a given time series.

        Parameters
        ----------
        node : _ProximityNode
            The current node in the tree.
        x : np.ndarray
            The time series to classify.
        der_x : np.ndarray
            The slope derivative of the time series, used for ddtw and wddtw.

        Returns
        -------
        class_distribution : np.ndarray
            The class distribution for the time series, i.e. the estimated probabilities
            for each class.
        """
        if node.is_leaf:
            return node.class_distribution
        else:
            exemplars, dist_name, dist_params = node.splitter

            # Use the slope derivative of the time series for ddtw and wddtw
            # use original distance to avoid recalculating every distance call
            x_used = x
            dist_used = dist_name
            if dist_name == "ddtw":
                x_used = der_x
                dist_used = "dtw"
            elif dist_name == "wddtw":
                x_used = der_x
                dist_used = "wdtw"

            min_dist = np.inf
            best_exemplar_label = None
            for label, exemplar in exemplars.items():
                dist = distance(
                    x_used,
                    exemplar,
                    method=dist_used,
                    **dist_params,
                )
                if dist < min_dist:
                    min_dist = dist
                    best_exemplar_label = label

            return self._traverse_tree(node.children[best_exemplar_label], x, der_x)

    def _get_candidate_splitter(self, X, der_X, rng, cls_idx, X_std):
        """Generate candidate splitter.

        Takes a time series dataset and a set of parameterized
        distance measures to create a candidate splitter, which
        contains a parameterized distance measure and a set of exemplars.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        der_X : np.ndarray shape (n_cases, n_channels, n_timepoints)
            The slope derivative of the training input samples, used for ddtw and wddtw.
        rng : np.random.RandomState
            Random number generator.
        cls_idx : dict
            The indices for each class label in the training set.
        X_std : float or None
            The standard deviation of the training set. If None, it will be
            calculated from the training set.

        Returns
        -------
        exemplars: dict
            The exemplars for each class, where the key is the class label
            and the value is the exemplar time series.
        dist_name: str
            The distance measure used for the splitter.
        dist_params: dict
            The parameters for the distance measure.
        X_std: float or None
            The standard deviation of the training set if calculated, otherwise same
            as input.
        """
        n = rng.randint(0, 11)
        derivative = False
        if n == 0:
            dist_name = "euclidean"
            dist_params = {}
        elif n == 1:
            dist_name = "dtw"
            dist_params = {}
        elif n == 2:
            dist_name = "dtw"
            dist_params = {"window": rng.uniform(0, 0.25)}
        elif n == 3:
            dist_name = "ddtw"
            dist_params = {}
            derivative = True
        elif n == 4:
            dist_name = "ddtw"
            dist_params = {"window": rng.uniform(0, 0.25)}
            derivative = True
        elif n == 5:
            dist_name = "wdtw"
            dist_params = {"g": rng.uniform(0, 1)}
        elif n == 6:
            dist_name = "wddtw"
            dist_params = {"g": rng.uniform(0, 1)}
            derivative = True
        elif n == 7:
            if X_std is None:
                X_std = self._get_std(X)
            dist_name = "erp"
            dist_params = {
                "window": rng.uniform(0, 0.25),
                "g": rng.uniform(X_std / 5, X_std),
            }
        elif n == 8:
            if X_std is None:
                X_std = self._get_std(X)
            dist_name = "lcss"
            dist_params = {
                "window": rng.uniform(0, 0.25),
                "epsilon": rng.uniform(X_std / 5, X_std),
            }
        elif n == 9:
            dist_name = "msm"
            dist_params = {"c": msm_params[rng.randint(0, len(msm_params))]}
        elif n == 10:
            dist_name = "twe"
            dist_params = {
                "nu": twe_nu_params[rng.randint(0, len(twe_nu_params))],
                "lmbda": twe_lmbda_params[rng.randint(0, len(twe_lmbda_params))],
            }
        else:
            raise ValueError(f"Invalid distance index {n}. Must be in range [0, 10].")

        exemplars = {}
        for label in cls_idx.keys():
            label_idx = cls_idx[label]
            id = rng.choice(label_idx)
            exemplars[label] = der_X[id] if derivative else X[id]

        return exemplars, dist_name, dist_params, X_std

    @staticmethod
    def _get_derivatives(X):
        if isinstance(X, np.ndarray):
            return slope_derivative_3d(X)
        else:
            der_X = []
            for x in X:
                der_X.append(slope_derivative_2d(x))
            return der_X

    @staticmethod
    def _get_std(X):
        if isinstance(X, np.ndarray):
            return std(X.flatten())
        else:
            return std(np.concatenate(X, axis=1).flatten())

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict | list[dict]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            ElasticEnsemble provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {
            "n_splitters": 3,
        }


msm_params = [
    0.01,
    0.01375,
    0.0175,
    0.02125,
    0.025,
    0.02875,
    0.0325,
    0.03625,
    0.04,
    0.04375,
    0.0475,
    0.05125,
    0.055,
    0.05875,
    0.0625,
    0.06625,
    0.07,
    0.07375,
    0.0775,
    0.08125,
    0.085,
    0.08875,
    0.0925,
    0.09625,
    0.1,
    0.136,
    0.172,
    0.208,
    0.244,
    0.28,
    0.316,
    0.352,
    0.388,
    0.424,
    0.46,
    0.496,
    0.532,
    0.568,
    0.604,
    0.64,
    0.676,
    0.712,
    0.748,
    0.784,
    0.82,
    0.856,
    0.892,
    0.928,
    0.964,
    1,
    1.36,
    1.72,
    2.08,
    2.44,
    2.8,
    3.16,
    3.52,
    3.88,
    4.24,
    4.6,
    4.96,
    5.32,
    5.68,
    6.04,
    6.4,
    6.76,
    7.12,
    7.48,
    7.84,
    8.2,
    8.56,
    8.92,
    9.28,
    9.64,
    10,
    13.6,
    17.2,
    20.8,
    24.4,
    28,
    31.6,
    35.2,
    38.8,
    42.4,
    46,
    49.6,
    53.2,
    56.8,
    60.4,
    64,
    67.6,
    71.2,
    74.8,
    78.4,
    82,
    85.6,
    89.2,
    92.8,
    96.4,
    100,
]

twe_nu_params = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

twe_lmbda_params = [
    0,
    0.011111111,
    0.022222222,
    0.033333333,
    0.044444444,
    0.055555556,
    0.066666667,
    0.077777778,
    0.088888889,
    0.1,
]
