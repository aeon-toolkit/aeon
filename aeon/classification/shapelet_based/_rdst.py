"""Random Dilated Shapelet Transform (RDST) Classifier.

A Random Dilated Shapelet Transform classifier pipeline that simply performs a random
shapelet dilated transform and builds (by default) a ridge classifier on the output.
"""

__maintainer__ = ["baraline"]
__all__ = ["RDSTClassifier"]

from typing import Union

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)


class RDSTClassifier(BaseClassifier):
    """
    A random dilated shapelet transform (RDST) classifier.

    Implementation of the random dilated shapelet transform classifier pipeline
    along the lines of [1]_, [2]_. Transforms the data using the
    `RandomDilatedShapeletTransform` and then builds a `RidgeClassifierCV` classifier
    with standard scaling.

    Parameters
    ----------
    max_shapelets : int, default=10000
        The maximum number of shapelets to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity has discarded the
        whole dataset.
    shapelet_lengths : array, default=None
        The set of possible lengths for shapelets. Each shapelet length is uniformly
        drawn from this set. If None, the shapelet length will be equal to
        min(max(2,n_timepoints//2),11).
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicates the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array, default=None
        The two perceniles used to select the threshold used to compute the Shapelet
        Occurrence feature. If None, the 5th and the 10th percentiles (i.e. [5,10])
        will be used.
    alpha_similarity : float, default=0.5
        The strength of the alpha similarity pruning. The higher the value, the fewer
        common indexes with previously sampled shapelets are allowed when sampling a
        new candidate with the same dilation parameter. It can cause the number of
        sampled shapelets to be lower than max_shapelets if the whole search space has
        been covered. The default is 0.5, and the maximum is 1. Values above it have
        no effect for now.
    use_prime_dilations : bool, default=False
        If True, restricts the value of the shapelet dilation parameter to be prime
        values. This can greatly speed-up the algorithm for long time series and/or
        short shapelet lengths, possibly at the cost of some accuracy.
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RidgeClassifierCV` classifier is used with standard scaling.
    save_transformed_data : bool, default=False
        If True, the transformed training dataset for all classifiers will be saved.
    class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
        Only applies if estimator is None, and the default is used.
        From sklearn documentation:
        If not given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))
        The “balanced_subsample” mode is the same as “balanced” except that weights
        are computed based on the bootstrap sample for every tree grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed through
        the fit method) if sample_weight is specified.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed training dataset for all classifiers. Only saved when
        ``save_transformed_data`` is `True`.

    See Also
    --------
    RandomDilatedShapeletTransform : The randomly dilated shapelet transform.
    RidgeClassifierCV : The default classifier used.

    References
    ----------
    .. [1] Antoine Guillaume et al. "Random Dilated Shapelet Transform: A New Approach
       for Time Series Shapelets", Pattern Recognition and Artificial Intelligence.
       ICPRAI 2022.
    .. [2] Antoine Guillaume, "Time series classification with shapelets: Application
       to predictive maintenance on event logs", PhD Thesis, University of Orléans,
       2023.


    Examples
    --------
    >>> from aeon.classification.shapelet_based import RDSTClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = RDSTClassifier(
    ...     max_shapelets=10
    ... )
    >>> clf.fit(X_train, y_train)
    RDSTClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:multithreading": True,
        "X_inner_type": ["np-list", "numpy3D"],
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        max_shapelets: int = 10000,
        shapelet_lengths=None,
        proba_normalization: float = 0.8,
        threshold_percentiles=None,
        alpha_similarity: float = 0.5,
        use_prime_dilations: bool = False,
        estimator=None,
        save_transformed_data: bool = False,
        class_weight=None,
        n_jobs: int = 1,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> None:
        self.max_shapelets = max_shapelets
        self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
        self.threshold_percentiles = threshold_percentiles
        self.alpha_similarity = alpha_similarity
        self.use_prime_dilations = use_prime_dilations
        self.estimator = estimator
        self.save_transformed_data = save_transformed_data
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.transformed_data_ = []

        self._transformer = None
        self._estimator = None

        super().__init__()

    def _fit(self, X, y):
        """Fit Classifier to training data.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The training input samples.
        y: array-like or list
            The class labels for samples in X.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        self._transformer = RandomDilatedShapeletTransform(
            max_shapelets=self.max_shapelets,
            shapelet_lengths=self.shapelet_lengths,
            proba_normalization=self.proba_normalization,
            threshold_percentiles=self.threshold_percentiles,
            alpha_similarity=self.alpha_similarity,
            use_prime_dilations=self.use_prime_dilations,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.estimator is None:
            self._estimator = make_pipeline(
                StandardScaler(with_mean=True),
                RidgeClassifierCV(
                    alphas=np.logspace(-4, 4, 20),
                    class_weight=self.class_weight,
                ),
            )
        else:
            self._estimator = _clone_estimator(self.estimator, self.random_state)
            m = getattr(self._estimator, "n_jobs", None)
            if m is not None:
                self._estimator.n_jobs = self.n_jobs

        X_t = self._transformer.fit_transform(X, y)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        self._estimator.fit(X_t, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        X_t = self._transformer.transform(X)

        return self._estimator.predict(X_t)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts label probabilities for sequences in X.

        Parameters
        ----------
        X: np.ndarray shape (n_cases, n_channels, n_timepoints)
            The data to predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        X_t = self._transformer.transform(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((len(X), self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, len(X)):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> Union[dict, list[dict]]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"max_shapelets": 20}
