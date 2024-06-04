import numpy as np
from sklearn.exceptions import NotFittedError

from aeon.classification.base import BaseClassifier
from aeon.distances import distance


class Node:

    def __init__(
        self,
        node_id,
        _is_leaf,
        label=None,
        splitter=None,
    ):
        self.node_id = node_id
        self._is_leaf = _is_leaf
        self.label = label
        self.splitter = splitter
        self.children = {}


class ProximityTree(BaseClassifier):

    def __init__(
        self,
        n_splitters: int = 5,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: int = 0,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        self.n_splitters = n_splitters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def get_parameter_value(self, X=None):
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
                param: np.round(np.random.uniform(low, high), 3)
                for param, (low, high) in ranges.items()
            }
        # For TWE
        lmbda = np.random.randint(0, 9)
        exponent_range = np.arange(1, 6)  # Exponents from -5 to 1 (inclusive)
        random_exponent = np.random.choice(exponent_range)
        nu = 1 / 10**random_exponent
        random_params["twe"] = {"lmbda": lmbda, "nu": nu}

        # For MSM
        base = 10
        # Exponents from -2 to 2 (inclusive)
        exponents = np.arange(-2, 3, dtype=np.float64)
        # Randomly select an index from the exponent range
        random_index = np.random.randint(0, len(exponents))
        c = base ** exponents[random_index]
        random_params["msm"] = {"c": c}

        return random_params

    def get_candidate_splitter(self, X, y):
        """Generate candidate splitter.

        Takes a time series dataset and a set of parameterized
        distance measures to create a candidate splitter, which
        contains a parameterized distance measure and a set of exemplars.

        Parameters
        ----------
        X : np.ndarray shape (n_cases, n_timepoints)
            The training input samples.
        y : np.array shape (n_cases,) or (n_cases,1)
        parameterized_distances : dictionary
            Contains the distances and their parameters.

        Returns
        -------
        splitter : list of two dictionaries
            A distance and its parameter values and a set of exemplars.
        """
        _X = X
        _y = y

        exemplars = {}
        for label in np.unique(_y):
            y_new = _y[_y == label]
            X_new = _X[_y == label]
            id = np.random.randint(0, X_new.shape[0])
            exemplars[y_new[id]] = X_new[id, :]

        # Create a list with first element exemplars and second element a
        # random parameterized distance measure
        parameterized_distances = self.get_parameter_value(X)
        n = np.random.randint(0, 9)
        dist = list(parameterized_distances.keys())[n]
        splitter = [exemplars, {dist: parameterized_distances[dist]}]

        return splitter

    def gini(self, y):
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
            unique_class_labels, class_counts = np.unique(y, return_counts=True)
            # subtract class entropy from current score for each class
            class_counts = np.divide(class_counts, n_instances)
            class_counts = np.power(class_counts, 2)
            sum = np.sum(class_counts)
            return 1 - sum
        else:
            # y is empty, therefore considered pure
            raise ValueError("y empty")

    def gini_gain(self, y, y_subs):
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
        if y.ndim != 1:
            raise ValueError()
        # find number of instances overall
        parent_n_instances = y.shape[0]
        # if parent has no instances then is pure
        if parent_n_instances == 0:
            for child in y_subs:
                if len(child) > 0:
                    raise ValueError("children populated but parent empty")
            return 0.5
        # find gini for parent node
        score = self.gini(y)
        # sum the children's gini scores
        for index in range(len(y_subs)):
            child_class_labels = y_subs[index]
            # ignore empty children
            if len(child_class_labels) > 0:
                # find gini score for this child
                child_score = self.gini(child_class_labels)
                # weight score by proportion of instances at child compared to
                # parent
                child_size = len(child_class_labels)
                child_score *= child_size / parent_n_instances
                # add to cumulative sum
                score -= child_score
        return score

    def _build_tree(self, X, y, depth, node_id, parent_target_value=None):

        # If the data reaching the node is empty
        if len(X) == 0:
            leaf_label = parent_target_value
            leaf = Node(node_id=node_id, _is_leaf=True, label=leaf_label)

        # Target value in current node
        target_value = self._find_target_value(y)

        # If max depth is reached
        if (self.max_depth is not None) and (depth >= self.max_depth):
            leaf_label = target_value
            leaf = Node(node_id=node_id, _is_leaf=True, label=leaf_label)

        # Pure node
        if len(np.unique(y)) == 1:
            leaf_label = target_value
            leaf = Node(node_id=node_id, _is_leaf=True, label=leaf_label)
            return leaf

        # Find the best splitter
        splitter = self.get_best_splitter(X, y)

        # Create root node
        node = Node(node_id=node_id, _is_leaf=False, splitter=splitter)

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
                    metric=measure,
                    kwargs=splitter[1][measure],
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

    def _find_target_value(self, y):
        """Get the class label of highest frequency."""
        unique, counts = np.unique(y, return_counts=True)
        # Find the index of the maximum count
        max_index = np.argmax(counts)
        mode_value = unique[max_index]
        # mode_count = counts[max_index]
        return mode_value

    def get_best_splitter(self, X, y):
        max_gain = float("-inf")
        best_splitter = None
        for _ in range(self.n_splitters):
            splitter = self.get_candidate_splitter(X, y)
            labels = list(splitter[0].keys())
            measure = list(splitter[1].keys())[0]
            y_subs = [[] for k in range(len(labels))]
            for j in range(X.shape[0]):
                min_dist = float("inf")
                sub = None
                for k in range(len(labels)):
                    dist = distance(
                        X[j],
                        splitter[0][labels[k]],
                        metric=measure,
                        kwargs=splitter[1][measure],
                    )
                    if dist < min_dist:
                        min_dist = dist
                        sub = k
                y_subs[sub].append(y[j])
            y_subs = [np.array(ele) for ele in y_subs]
            gini_index = self.gini_gain(y, y_subs)
            if gini_index > max_gain:
                max_gain = gini_index
                best_splitter = splitter
        return best_splitter

    def _fit(self, X, y):
        self.root = self._build_tree(
            X, y, depth=0, node_id="0", parent_target_value=None
        )
        self._is_fitted = True

    def _predict(self, X):
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        predictions = list()
        for i in range(len(X)):
            prediction = self.classify(self.root, X[i])
            predictions.append(prediction)
        return predictions

    def classify(self, treenode, x):
        # classify one datapoint using the proximity tree
        if treenode._is_leaf is True:
            return treenode.label
        else:
            return self.classify(treenode.children, x)

    def _predict_proba(self, X):
        pass
