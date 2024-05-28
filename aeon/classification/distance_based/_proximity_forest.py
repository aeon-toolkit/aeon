import numpy as np

from aeon.classification.base import BaseClassifier

# from aeon.distances import distance


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
        self.n_splitter = n_splitters
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

    def get_candidate_splitter(self, X, y, parameterized_distances):
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
        n = np.random.randint(0, 9)
        dist = list(parameterized_distances.keys())[n]
        splitter = [exemplars, {dist: parameterized_distances[dist]}]

        return splitter

    def _is_leaf(self, y):
        if len(np.unique(y)) > 1:
            return False
        return True

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
        y : 1d array like
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
        y = np.array(y)
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

    def _fit(self, X, y):
        pass

    def _predict(self, X):
        pass

    def _predict_proba(self, X):
        pass
