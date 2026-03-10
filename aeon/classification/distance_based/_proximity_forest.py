"""Proximity Forest Classifier.

The Proximity Forest is an ensemble of Proximity Trees.
"""

__maintainer__ = []
__all__ = ["ProximityForest"]


import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based._proximity_tree import ProximityTree
from aeon.utils.validation import check_n_jobs


class ProximityForest(BaseClassifier):
    """Proximity Forest Classifier.

    The Proximity Forest is a distance-based classifier that creates an
    ensemble of decision trees, where the splits are based on the
    similarity between time series measured using various parameterised
    distance measures.

    Parameters
    ----------
    n_trees: int, default = 100
        The number of trees, by default an ensemble of ``100`` trees is formed.
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
    n_jobs : int, default = 1
        The number of parallel jobs to run for neighbors search.
        ``None`` means ``1`` unless in a :obj:``joblib.parallel_backend`` context.
        ``-1`` means using all processors.
        for more details. Parameter for compatibility purposes, still unimplemented.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in ``joblib``, if ``None``
        a ``prefer`` value of ``"threads"`` is used by default.
        Valid options are ``"loky"``, ``"multiprocessing"``, ``"threading"`` or a
        custom backend.
        See the ``joblib`` Parallel documentation for more details.

    Notes
    -----
    For the Java version, see
    `ProximityForest
    <https://github.com/fpetitjean/ProximityForest>`_.

    References
    ----------
    .. [1] Lucas, B., Shifaz, A., Pelletier, C., O’Neill, L., Zaidi, N., Goethals, B.,
    Petitjean, F. and Webb, G.I., 2019. Proximity forest: an effective and scalable
    distance-based classifier for time series. Data Mining and Knowledge Discovery,
    33(3), pp.607-635.

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.distance_based import ProximityForest
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> classifier = ProximityForest(n_trees=10, n_splitters=3)
    >>> classifier.fit(X_train, y_train)
    ProximityForest(...)
    >>> y_pred = classifier.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "algorithm_type": "distance",
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        n_trees=100,
        n_splitters: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int | np.random.RandomState | None = None,
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
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.trees_ = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._fit_tree)(
                X,
                y,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for _ in range(self.n_trees)
        )

    def _predict(self, X):
        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in self._predict_proba(X)]
        )

    def _predict_proba(self, X):
        preds = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(delayed(self._predict_tree)(X, tree) for tree in self.trees_)

        votes = np.zeros((len(X), self.n_classes_))
        for i in range(len(preds)):
            for j in range(len(X)):
                votes[j, self._class_dictionary[preds[i][j]]] += 1

        return votes / self.n_trees

    def _fit_tree(self, X, y, random_state):
        clf = ProximityTree(
            n_splitters=self.n_splitters,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=random_state,
        )
        clf.fit(X, y)
        return clf

    def _predict_tree(self, X, tree):
        return tree.predict(X)

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
        return {"n_trees": 3, "n_splitters": 3, "max_depth": 3}
