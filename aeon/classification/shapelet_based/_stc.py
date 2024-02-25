"""A shapelet transform classifier (STC).

Shapelet transform classifier pipeline that simply performs a (configurable) shapelet
transform then builds (by default) a rotation forest classifier on the output.
"""

__maintainer__ = []
__all__ = ["ShapeletTransformClassifier"]

import warnings

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform


class ShapeletTransformClassifier(BaseClassifier):
    """
    A shapelet transform classifier (STC).

    Implementation of the binary shapelet transform classifier pipeline along the lines
    of [1]_[2]_ but with random shapelet sampling. Transforms the data using the
    configurable `RandomShapeletTransform` and then builds a `RotationForestClassifier`
    classifier.

    As some implementations and applications contract the transformation solely,
    contracting is available for the transform only and both classifier and transform.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to ``<= max_shapelets``, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to ``n_classes_ / max_shapelets``. If `None`, uses the
        minimum between ``10 * n_instances_`` and `1000`.
    max_shapelet_length : int or None, default=None
        Lower bound on candidate shapelet lengths for the transform. If ``None``, no
        max length is used
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RotationForestClassifier` classifier is used.
    transform_limit_in_minutes : int, default=0
        Time contract to limit transform time in minutes for the shapelet transform,
        overriding `n_shapelet_samples`. A value of `0` means ``n_shapelet_samples``
        is used.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding ``n_shapelet_samples``
        and ``transform_limit_in_minutes``. The ``estimator`` will only be contracted if
        a ``time_limit_in_minutes parameter`` is present. Default of `0` means
        ``n_shapelet_samples`` or ``transform_limit_in_minutes`` is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when contracting the transform with
        ``transform_limit_in_minutes`` or ``time_limit_in_minutes``.
    save_transformed_data : bool, default="deprecated"
        Save the data transformed in ``fit``.

        Deprecated and will be removed in v0.8.0. Use ``fit_predict`` and
        ``fit_predict_proba`` to generate train estimates instead.
        ``transformed_data_`` will also be removed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets in the transform.
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
    fit_time_  : int
        The time (in milliseconds) for ``fit`` to run.
    n_instances_ : int
        The number of train cases in the training set.
    n_dims_ : int
        The number of dimensions per case in the training set.
    series_length_ : int
        The length of each series in the training set.

    See Also
    --------
    RandomShapeletTransform : The randomly sampled shapelet transform.
    RotationForestClassifier : The default rotation forest classifier used.

    Notes
    -----
    For the Java version, see
    `tsml <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/classifiers/shapelet_based/ShapeletTransformClassifier.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.

    Examples
    --------
    >>> from aeon.classification.shapelet_based import ShapeletTransformClassifier
    >>> from aeon.classification.sklearn import RotationForestClassifier
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = ShapeletTransformClassifier(
    ...     estimator=RotationForestClassifier(n_estimators=3),
    ...     n_shapelet_samples=100,
    ...     max_shapelets=10,
    ...     batch_size=20,
    ... )
    >>> clf.fit(X_train, y_train)
    ShapeletTransformClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "shapelet",
    }

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        max_shapelet_length=None,
        estimator=None,
        transform_limit_in_minutes=0,
        time_limit_in_minutes=0,
        contract_max_n_shapelet_samples=np.inf,
        save_transformed_data="deprecated",
        n_jobs=1,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.estimator = estimator

        self.transform_limit_in_minutes = transform_limit_in_minutes
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples

        self.random_state = random_state
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0

        self._transformer = None
        self._estimator = estimator
        self._transform_limit_in_minutes = 0
        self._classifier_limit_in_minutes = 0

        # TODO remove 'save_transformed_data' and 'transformed_data_' in v0.8.0
        self.transformed_data_ = []
        self.save_transformed_data = save_transformed_data
        if save_transformed_data != "deprecated":
            warnings.warn(
                "the save_transformed_data parameter is deprecated and will be"
                "removed in v0.8.0. transformed_data_ will also be removed.",
                stacklevel=2,
            )

        super().__init__()

    def _fit(self, X, y):
        """Fit ShapeletTransformClassifier to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        b = (
            False
            if isinstance(self.save_transformed_data, str)
            else self.save_transformed_data
        )
        self.transformed_data_ = self._fit_stc(X, y)
        if not b:
            self.transformed_data_ = []
        return self

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        X_t = self._transformer.transform(X)

        return self._estimator.predict(X_t)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        X_t = self._transformer.transform(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._fit_predict_proba(X, y)
            ]
        )

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        Xt = self._fit_stc(X, y)

        if (isinstance(self.estimator, RotationForestClassifier)) or (
            self.estimator is None
        ):
            return self._estimator._get_train_probs(Xt, y)
        else:
            m = getattr(self._estimator, "predict_proba", None)
            if not callable(m):
                raise ValueError("Estimator must have a predict_proba method.")

            cv_size = 10
            _, counts = np.unique(y, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class
                if cv_size < 2:
                    raise ValueError(
                        "All classes must have at least 2 values to run the "
                        "fit_predict/fit_predict_proba cross-validation."
                    )

            estimator = _clone_estimator(self.estimator, self.random_state)

            return cross_val_predict(
                estimator,
                X=Xt,
                y=y,
                cv=cv_size,
                method="predict_proba",
                n_jobs=self._n_jobs,
            )

    def _fit_stc(self, X, y):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.time_limit_in_minutes > 0:
            # contracting 2/3 transform (with 1/5 of that taken away for final
            # transform), 1/3 classifier
            third = self.time_limit_in_minutes / 3
            self._classifier_limit_in_minutes = third
            self._transform_limit_in_minutes = (third * 2) / 5 * 4
        elif self.transform_limit_in_minutes > 0:
            self._transform_limit_in_minutes = self.transform_limit_in_minutes

        self._transformer = RandomShapeletTransform(
            n_shapelet_samples=self.n_shapelet_samples,
            max_shapelets=self.max_shapelets,
            max_shapelet_length=self.max_shapelet_length,
            time_limit_in_minutes=self._transform_limit_in_minutes,
            contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            RotationForestClassifier() if self.estimator is None else self.estimator,
            self.random_state,
        )

        if isinstance(self._estimator, RotationForestClassifier):
            self._estimator.save_transformed_data = self.save_transformed_data

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        m = getattr(self._estimator, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes

        Xt = self._transformer.fit_transform(X, y)

        self._estimator.fit(Xt, y)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            ShapeletTransformClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contacting
                    functionality
                "train_estimate" - used in some classifiers that set the
                    "capability:train_estimate" tag to True to allow for more efficient
                    testing when relevant parameters are available

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sklearn.ensemble import RandomForestClassifier

        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=5),
                "n_shapelet_samples": 50,
                "max_shapelets": 10,
                "batch_size": 10,
            }
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "estimator": RotationForestClassifier(contract_max_n_estimators=2),
                "contract_max_n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
            }
        else:
            return {
                "estimator": RotationForestClassifier(n_estimators=2),
                "n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
            }
