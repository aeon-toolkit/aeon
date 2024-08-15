"""Proximity Forest 2.0 Classifier."""

from typing import Type, Union

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numba.typed import List as NumbaList
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based._distances import distance
from aeon.distances import minkowski_distance


class ProximityForest2(BaseClassifier):
    """Proximity Forest 2.0 Classifier.

    The Proximity Forest 2.0 outperforms the Proximity Forest Classifier.
    PF 2.0 incorporates three recent advances in time series similarity measures
    (1) computationally e cient early abandoning and pruning to speedup elastic
    similarity computations; (2) a new elastic similarity measure, Amerced Dynamic
    Time Warping (ADTW); and (3) cost function tuning.

    Parameters
    ----------
    n_trees: int, default = 100
        The number of trees, by default an ensemble of 100 trees is formed.
    n_splitters: int, default = 5
        The number of candidate splitters to be evaluated at each node.
    max_depth: int, default = None
        The maximum depth of the tree. If None, then nodes are expanded until all
        leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split: int, default = 2
        The minimum number of samples required to split an internal node.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default = 1
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details. Parameter for compatibility purposes, still unimplemented.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Notes
    -----
    For the C++ version, see
    `ProximityForest2.0
    <https://github.com/MonashTS/ProximityForest2.0>`_.

    Also refer to the Proximity Forest implementation,
    see the :class:`~aeon.classification.distance_based.ProximityForest` API.

    References
    ----------
    .. [1] Matthieu Herrmann, Chang Wei Tan, Mahsa Salehi, Geoffrey I. Webb.
    Proximity Forest 2.0: A new e ective and scalable similarity-based classifier
    for time series, https://doi.org/10.48550/arXiv.2304.05800.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import ProximityForest
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = ProximityForest2(n_trees = 10, n_splitters = 3)
    >>> classifier.fit(X_train, y_train)
    ProximityForest2(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:multithreading": True,
        "algorithm_type": "distance",
        "X_inner_type": ["numpy2D"],
    }

    def __init__(
        self,
        n_trees=100,
        n_splitters: int = 5,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: Union[int, Type[np.random.RandomState], None] = None,
        n_jobs: int = 1,
        parallel_backend=None,
    ):
        self.n_trees = n_trees
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        super().__init__()

    def _fit(self, X, y):
        rng = check_random_state(self.random_state)
        self.trees_ = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(_fit_tree)(
                X,
                y,
                self.n_splitters,
                self.max_depth,
                self.min_samples_split,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for _ in range(self.n_trees)
        )

    def _predict_proba(self, X):
        classes = list(self.classes_)
        preds = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(delayed(_predict_tree)(tree, X) for tree in self.trees_)
        n_cases = X.shape[0]
        votes = np.zeros((n_cases, self.n_classes_))
        for i in range(len(preds)):
            predictions = np.array(
                [classes.index(class_label) for class_label in preds[i]]
            )
            for j in range(n_cases):
                votes[j, predictions[j]] += 1
        output_probas = votes / self.n_trees
        return output_probas

    def _predict(self, X):
        probas = self._predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds


def _fit_tree(X, y, n_splitters, max_depth, min_samples_split, random_state):
    clf = ProximityTree2(
        n_splitters=n_splitters,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def _predict_tree(tree, X):
    return tree.predict(X)


class _Node:
    """Proximity Tree node.

    Parameters
    ----------
    node_id: str
        The id of node, root node has id 0.
    _is_leaf: bool
        To identify leaf nodes.
    label: int, str or None
        Contains the class label of leaf node, None otherwise.
    splitter: dict
        The splitter used to split the node.
    class_distribution: dict
        In case of unpure leaf node, save the class distribution to calculate
        probability of each class.
    children: dict
        Contains the class label and the associated node, empty for leaf node.
    """

    def __init__(
        self,
        node_id: str,
        _is_leaf: bool,
        label=None,
        class_distribution=None,
        splitter=None,
    ):
        self.node_id = node_id
        self._is_leaf = _is_leaf
        self.label = label
        self.splitter = splitter
        self.class_distribution = class_distribution or {}
        self.children = {}


class ProximityTree2(BaseClassifier):
    """Proximity Tree classifier for PF2.0.

    It leverages ADTW and tuning the cost functions for TSC to increase accuracy,
    together with the computational e ciency of computing these distances provided by
    EAP.

    PF 2.0 modifies the splitters to comprise three elements, a set of class exemplars,
    a parameterised similarity measure and a time series transform.

    Parameters
    ----------
    n_splitters: int, default = 5
        The number of candidate splitters to be evaluated at each node.
    max_depth: int, default = None
        The maximum depth of the tree. If None, then nodes are expanded until all
        leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split: int, default = 2
        The minimum number of samples required to split an internal node.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Notes
    -----
    For the C++ version, see
    `ProximityForest2.0
    <https://github.com/MonashTS/ProximityForest2.0>`_.

    Also refer to the Proximity Tree implementation,
    see the :class:`~aeon.classification.distance_based.ProximityTree` API.

    References
    ----------
    .. [1] Matthieu Herrmann, Chang Wei Tan, Mahsa Salehi, Geoffrey I. Webb.
    Proximity Forest 2.0: A new e ective and scalable similarity-based classifier
    for time series, https://doi.org/10.48550/arXiv.2304.05800.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import ProximityTree
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = ProximityTree2(n_splitters = 3)
    >>> classifier.fit(X_train, y_train)
    ProximityTree2(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "algorithm_type": "distance",
        "X_inner_type": ["numpy2D"],
    }

    def __init__(
        self,
        n_splitters: int = 5,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: Union[int, Type[np.random.RandomState], None] = None,
    ) -> None:
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        super().__init__()

    def _get_parameter_value(self, X):
        """Generate random parameter values.

        For a list of distance measures, generate a dictionary
        of parameterized distances.

        Parameters
        ----------
        X : np.ndarray of shape (n_cases, n_timepoints)

        Returns
        -------
        distance_param : a dictionary of distances and their
        parameters.
        """
        rng = check_random_state(self.random_state)

        X_std = X.std()
        param_ranges = {
            "dtw": {"window": (0, 0.25)},
            "adtw": {"window": (0, 0.25)},
            "lcss": {"epsilon": (X_std / 5, X_std), "window": (0, 0.25)},
        }
        random_params = {}
        for measure, ranges in param_ranges.items():
            random_params[measure] = {
                param: np.round(rng.uniform(low, high), 3)
                for param, (low, high) in ranges.items()
            }

        return random_params

    def _get_candidate_splitter(self, X, y):
        """Generate candidate splitter.

        Takes a time series dataset and a set of parameterized
        distance measures to create a candidate splitter, which
        contains a parameterized distance measure and a set of exemplars.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_timepoints)
            The training input samples.
        y : np.array shape (n_cases,) or (n_cases,1)
            The labels of the training samples.
        parameterized_distances : dictionary
            Contains the distances and their parameters.

        Returns
        -------
        splitter : list of two dictionaries
            A distance and its parameter values and a set of exemplars.
        """
        rng = check_random_state(self.random_state)

        # Class exemplars
        exemplars = {}
        for label in np.unique(y):
            y_new = y[y == label]
            X_new = X[y == label]
            id = rng.randint(0, X_new.shape[0])
            exemplars[y_new[id]] = X_new[id, :]

        # Time series transform
        transforms = ["raw", "first_derivative"]
        t = rng.randint(0, 2)
        transform = transforms[t]

        # random parameterized distance measure
        parameterized_distances = self._get_parameter_value(X)
        n = rng.randint(0, 3)
        dist = list(parameterized_distances.keys())[n]
        if dist == "dtw":
            values = [0.5, 1, 2]
            p = rng.choice(values)
            parameterized_distances[dist]["p"] = p
        if dist == "adtw":
            values = [0.5, 1, 2]
            p = rng.choice(values)
            parameterized_distances[dist]["p"] = p
            i = rng.randint(1, 101)
            w = ((i / 100) ** 5) * self._max_warp_penalty
            parameterized_distances[dist]["warp_penalty"] = w

        # Create a list of class exemplars, distance measures and transform
        splitter = [exemplars, {dist: parameterized_distances[dist]}, transform]

        return splitter

    def _get_best_splitter(self, X, y):
        """Get the splitter for a node which maximizes the gini gain."""
        max_gain = float("-inf")
        best_splitter = None
        for _ in range(self.n_splitters):
            splitter = self._get_candidate_splitter(X, y)
            labels = list(splitter[0].keys())
            measure = list(splitter[1].keys())[0]
            transform = splitter[2]
            if transform == "first_derivative":
                X_trans = NumbaList()
                for i in range(len(X)):
                    X_trans.append(first_order_derivative(X[i]))
                exemplars = NumbaList()
                for i in range(len(labels)):
                    exemplars.append(
                        first_order_derivative(list(splitter[0].values())[i])
                    )
            else:
                X_trans = X
                exemplars = np.array(list(splitter[0].values()))

            y_subs = [[] for _ in range(len(labels))]
            for j in range(len(X_trans)):
                min_dist = float("inf")
                sub = None
                for k in range(len(labels)):
                    dist = distance(
                        X_trans[j],
                        exemplars[k],
                        metric=measure,
                        **splitter[1][measure],
                    )
                    if dist < min_dist:
                        min_dist = dist
                        sub = k
                y_subs[sub].append(y[j])
            y_subs = [np.array(ele, dtype=y.dtype) for ele in y_subs]
            gini_index = gini_gain(y, y_subs)
            if gini_index > max_gain:
                max_gain = gini_index
                best_splitter = splitter
        return best_splitter

    def _build_tree(self, X, y, depth, node_id, parent_target_value=None):
        """Build the tree recursively from the root node down to the leaf nodes."""
        # If the data reaching the node is empty
        if len(X) == 0:
            leaf_label = parent_target_value
            leaf_distribution = {}
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=leaf_distribution,
            )
            return leaf

        # Target value in current node
        target_value = self._find_target_value(y)
        class_distribution = {
            label: count / len(y)
            for label, count in zip(*np.unique(y, return_counts=True))
        }

        # Pure node
        if len(np.unique(y)) == 1:
            leaf_label = target_value
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=class_distribution,
            )
            return leaf

        # If min sample splits is reached
        if self.min_samples_split >= len(X):
            leaf_label = target_value
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=class_distribution,
            )
            return leaf

        # If max depth is reached
        if (self.max_depth is not None) and (depth >= self.max_depth):
            leaf_label = target_value
            leaf = _Node(
                node_id=node_id,
                _is_leaf=True,
                label=leaf_label,
                class_distribution=class_distribution,
            )
            return leaf

        # Find the best splitter
        splitter = self._get_best_splitter(X, y)

        # Create root node
        node = _Node(node_id=node_id, _is_leaf=False, splitter=splitter)

        # For each exemplar split the data
        labels = list(splitter[0].keys())
        measure = list(splitter[1].keys())[0]
        transform = splitter[2]
        if transform == "first_derivative":
            X_trans = NumbaList()
            for i in range(len(X)):
                X_trans.append(first_order_derivative(X[i]))
            exemplars = NumbaList()
            for i in range(len(labels)):
                exemplars.append(first_order_derivative(list(splitter[0].values())[i]))
        else:
            X_trans = X
            exemplars = np.array(list(splitter[0].values()))

        X_child = [[] for _ in labels]
        y_child = [[] for _ in labels]
        for i in range(len(X_trans)):
            min_dist = np.inf
            id = None
            for j in range(len(labels)):
                dist = distance(
                    X_trans[i],
                    exemplars[j],
                    metric=measure,
                    **splitter[1][measure],
                )
                if dist < min_dist:
                    min_dist = dist
                    id = j
            X_child[id].append(X[i])
            y_child[id].append(y[i])
        X_child = [np.array(ele) for ele in X_child]
        y_child = [np.array(ele) for ele in y_child]
        # For each exemplar, create a branch
        for i in range(len(labels)):
            child_node_id = node_id + "." + str(i)
            child_node = self._build_tree(
                X_child[i],
                y_child[i],
                depth=depth + 1,
                node_id=child_node_id,
                parent_target_value=target_value,
            )
            node.children[labels[i]] = child_node

        return node

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _find_target_value(y):
        """Get the class label of highest frequency."""
        unique_labels = list(np.unique(y))
        class_counts = []
        for i in range(len(unique_labels)):
            cnt = 0
            for j in range(len(y)):
                if y[j] == unique_labels[i]:
                    cnt += 1
            class_counts.append(cnt)
        class_counts = np.array(class_counts)
        # Find the index of the maximum count
        max_index = np.argmax(class_counts)
        mode_value = unique_labels[max_index]
        # mode_count = counts[max_index]
        return mode_value

    def _fit(self, X, y):
        self._max_warp_penalty = _global_warp_penalty(X)
        self.root = self._build_tree(
            X, y, depth=0, node_id="0", parent_target_value=None
        )

    def _predict(self, X):
        probas = self._predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return np.array([self.classes_[pred] for pred in predictions])

    def _predict_proba(self, X):
        # Get the unique class labels
        classes = list(self.classes_)
        class_count = len(classes)
        probas = []

        for i in range(len(X)):
            # Classify the data point and find the leaf node
            leaf_node = self._classify(self.root, X[i])

            # Create probability distribution based on class counts in the leaf node
            proba = np.zeros(class_count)
            for class_label, class_proba in leaf_node.class_distribution.items():
                proba[classes.index(class_label)] = class_proba
            probas.append(proba)

        return np.array(probas)

    def _classify(self, treenode, x):
        # Classify one data point using the proximity tree
        if treenode._is_leaf:
            return treenode
        else:
            measure = list(treenode.splitter[1].keys())[0]
            branches = list(treenode.splitter[0].keys())
            transform = treenode.splitter[2]
            if transform == "first_derivative":
                x_trans = first_order_derivative(x)
                exemplars = NumbaList()
                for i in range(len(list(treenode.splitter[0].values()))):
                    exemplars.append(
                        first_order_derivative(list(treenode.splitter[0].values())[i])
                    )
            else:
                x_trans = x
                exemplars = np.array(list(treenode.splitter[0].values()))
            min_dist = np.inf
            id = None
            for i in range(len(branches)):
                dist = distance(
                    x_trans,
                    exemplars[i],
                    metric=measure,
                    **treenode.splitter[1][measure],
                )
                if dist < min_dist:
                    min_dist = dist
                    id = i
            return self._classify(treenode.children[branches[id]], x)


@njit(cache=True, fastmath=True)
def gini(y) -> float:
    """Get gini score at a specific node.

    Parameters
    ----------
    y : 1d numpy array
        array of class labels

    Returns
    -------
    score : float
        gini score for the set of class labels (i.e. how pure they are). A
        larger score means more impurity. Zero means
        pure.
    """
    # get number instances at node
    n_instances = y.shape[0]
    if n_instances > 0:
        # count each class
        unique_labels = list(np.unique(y))
        class_counts = []
        for i in range(len(unique_labels)):
            cnt = 0
            for j in range(len(y)):
                if y[j] == unique_labels[i]:
                    cnt += 1
            class_counts.append(cnt)
        class_counts = np.array(class_counts)
        # subtract class entropy from current score for each class
        class_counts = np.divide(class_counts, n_instances)
        class_counts = np.power(class_counts, 2)
        sum = np.sum(class_counts)
        return 1 - sum
    else:
        # y is empty, therefore considered pure
        raise ValueError("y empty")


@njit(cache=True, fastmath=True)
def gini_gain(y, y_subs) -> float:
    """Get gini score of a split, i.e. the gain from parent to children.

    Parameters
    ----------
    y : 1d array
        array of class labels at parent
    y_subs : list of 1d array like
        list of array of class labels, one array per child

    Returns
    -------
    score : float
        gini score of the split from parent class labels to children. Note a
        higher score means better gain,
        i.e. a better split
    """
    # find number of instances overall
    parent_n_instances = y.shape[0]
    # if parent has no instances then is pure
    if parent_n_instances == 0:
        for child in y_subs:
            if len(child) > 0:
                raise ValueError("children populated but parent empty")
        return 0.5
    # find gini for parent node
    score = gini(y)
    # sum the children's gini scores
    for index in range(len(y_subs)):
        child_class_labels = y_subs[index]
        # ignore empty children
        if len(child_class_labels) > 0:
            # find gini score for this child
            child_score = gini(child_class_labels)
            # weight score by proportion of instances at child compared to
            # parent
            child_size = len(child_class_labels)
            child_score *= child_size / parent_n_instances
            # add to cumulative sum
            score -= child_score
    return score


@njit(cache=True, fastmath=True)
def _global_warp_penalty(X):
    n = len(X)
    pairs = [
        (X[np.random.randint(0, n)], X[np.random.randint(0, n)]) for _ in range(4000)
    ]
    distances = np.array([minkowski_distance(pair[0], pair[1], 2) for pair in pairs])
    penalty = np.mean(distances)
    return penalty


@njit(cache=True, fastmath=True)
def first_order_derivative_1d(q: np.ndarray) -> np.ndarray:
    """Compute the first-order derivative of a 1D time series."""
    n = len(q)
    result = np.zeros(n)

    # First element
    result[0] = (q[1] - q[0]) / 2.0

    # Middle elements
    for j in range(1, n - 1):
        result[j] = ((q[j] - q[j - 1]) + (q[j + 1] - q[j - 1]) / 2.0) / 2.0

    # Last element
    result[-1] = (q[-1] - q[-2]) / 2.0

    return result


@njit(cache=True, fastmath=True)
def first_order_derivative_2d(q: np.ndarray) -> np.ndarray:
    """Compute the first-order derivative of a 2D time series."""
    n = q.shape[1]
    result = np.zeros((1, n))

    # First element
    result[0, 0] = (q[0, 1] - q[0, 0]) / 2.0

    # Middle elements
    for j in range(1, n - 1):
        result[0, j] = (
            (q[0, j] - q[0, j - 1]) + (q[0, j + 1] - q[0, j - 1]) / 2.0
        ) / 2.0

    # Last element
    result[0, -1] = (q[0, -1] - q[0, -2]) / 2.0

    return result


@njit(cache=True, fastmath=True)
def first_order_derivative(q: np.ndarray) -> np.ndarray:
    """
    Compute the first-order derivative of the time series.

    Parameters
    ----------
    q : np.ndarray (n_timepoints,) or (1, n_timepoints)
        Time series to take derivative of.

    Returns
    -------
    np.ndarray
        The first-order derivative of q with the same shape as the input.
    """
    if q.ndim == 1:
        return first_order_derivative_1d(q)
    elif q.ndim == 2 and q.shape[0] == 1:
        return first_order_derivative_2d(q)
    else:
        raise ValueError(
            "Time series must be either (n_timepoints,) or (1, n_timepoints)."
        )
