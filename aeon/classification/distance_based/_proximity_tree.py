"""Proximity Tree Time Series Classifier.

A decision tree classifier where the splits based on the
similarity of instances to chosen time series exemplars, measured using
aeon distances.
"""

from typing import Optional, Union

import numpy as np
from numba import njit
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.distances import distance


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
    >>> classifier = ProximityTree(n_splitters = 3)
    >>> classifier.fit(X_train, y_train)
    ProximityTree(...)
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
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Union[int, np.random.RandomState, None] = None,
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
            "euclidean": {},
            "dtw": {"window": (0, 0.25)},
            "ddtw": {"window": (0, 0.25)},
            "wdtw": {"g": (0, 1)},
            "wddtw": {"g": (0, 1)},
            "erp": {"g": (X_std / 5, X_std)},
            "lcss": {"epsilon": (X_std / 5, X_std), "window": (0, 0.25)},
        }
        random_params = {}
        for measure, ranges in param_ranges.items():
            random_params[measure] = {
                param: np.round(rng.uniform(low, high), 3)
                for param, (low, high) in ranges.items()
            }
        # For TWE
        lmbda = rng.randint(0, 9)
        exponent_range = np.arange(1, 6)  # Exponents from -5 to 1 (inclusive)
        random_exponent = rng.choice(exponent_range)
        nu = 1 / 10**random_exponent
        random_params["twe"] = {"lmbda": lmbda, "nu": nu}

        # For MSM
        base = 10
        # Exponents from -2 to 2 (inclusive)
        exponents = np.arange(-2, 3, dtype=np.float64)
        # Randomly select an index from the exponent range
        random_index = rng.randint(0, len(exponents))
        c = base ** exponents[random_index]
        random_params["msm"] = {"c": c}

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

        exemplars = {}
        for label in np.unique(y):
            y_new = y[y == label]
            X_new = X[y == label]
            id = rng.randint(0, X_new.shape[0])
            exemplars[y_new[id]] = X_new[id, :]

        # Create a list with first element exemplars and second element a
        # random parameterized distance measure
        parameterized_distances = self._get_parameter_value(X)
        n = rng.randint(0, 9)
        dist = list(parameterized_distances.keys())[n]
        splitter = [exemplars, {dist: parameterized_distances[dist]}]

        return splitter

    def _get_best_splitter(self, X, y):
        """Get the splitter for a node which maximizes the gini gain."""
        max_gain = float("-inf")
        best_splitter = None
        for _ in range(self.n_splitters):
            splitter = self._get_candidate_splitter(X, y)
            labels = list(splitter[0].keys())
            measure = list(splitter[1].keys())[0]
            y_subs = [[] for _ in range(len(labels))]
            for j in range(X.shape[0]):
                min_dist = float("inf")
                sub = None
                for k in range(len(labels)):
                    dist = distance(
                        X[j],
                        splitter[0][labels[k]],
                        method=measure,
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
        X_child = [[] for _ in labels]
        y_child = [[] for _ in labels]
        for i in range(len(X)):
            min_dist = np.inf
            id = None
            for j in range(len(labels)):
                dist = distance(
                    X[i],
                    splitter[0][labels[j]],
                    method=measure,
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
            min_dist = np.inf
            id = None
            for i in range(len(branches)):
                dist = distance(
                    x,
                    treenode.splitter[0][branches[i]],
                    method=measure,
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
