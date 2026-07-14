"""Shapelet Transform Classifier (STC).

Pipeline combining a configurable shapelet transform with a tabular classifier.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ShapeletTransformClassifier"]


from time import perf_counter

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from aeon.utils.validation import check_n_jobs


class ShapeletTransformClassifier(BaseClassifier):
    """Shapelet Transform Classifier (STC).

    STC transforms time series into distances from discriminative subsequences using
    ``RandomShapeletTransform``, then fits a classifier to the transformed data. The
    default classifier is ``RotationForestClassifier``. This implementation follows
    [1]_ and [2]_, with random candidate-shapelet sampling.

    The shapelet transform can be contracted independently, or a total contract can be
    divided between the transform and an estimator that supports
    ``time_limit_in_minutes``.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        Number of candidate shapelets assessed by the transform. The transform retains
        at most ``max_shapelets`` candidates with the highest information gain.
    max_shapelets : int or None, default=None
        Maximum number of shapelets retained by the transform. The budget is divided
        equally among classes. If None, use the smaller of 10 times the number of
        training cases and 1000.
    max_shapelet_length : int or None, default=None
        Upper bound on candidate shapelet lengths. If None, use the length of the
        longest training series.
    estimator : BaseEstimator or None, default=None
        Classifier fitted to the transformed data. Must implement the scikit-learn
        estimator interface. If None, use ``RotationForestClassifier``.
    batch_size : int, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets in the transform.
    verbose : int, default=0
        Level of output printed to the console. ``0`` prints no output, ``1``
        prints STC phase timings and component progress, and ``2`` or greater
        prints detailed progress from the shapelet transform and estimator.
    transform_limit_in_minutes : float, default=0
        Independent time contract for fitting the shapelet transform, in minutes,
        overriding ``n_shapelet_samples``. A value of 0 uses
        ``n_shapelet_samples``.
    time_limit_in_minutes : float, default=0
        Total time contract for fitting STC, in minutes. A positive value overrides
        ``transform_limit_in_minutes`` and divides time between the transform and the
        classifier. The classifier is contracted only if it exposes a
        ``time_limit_in_minutes`` parameter.
    contract_max_n_shapelet_samples : int or float, default=np.inf
        Maximum number of candidate shapelets assessed when contracting the
        transform via ``transform_limit_in_minutes`` or ``time_limit_in_minutes``.
    n_jobs : int, default=1
        The number of jobs used by the transform and classifier. ``-1`` uses all
        processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes_)
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.

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
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
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
        "capability:unequal_length": True,
        "algorithm_type": "shapelet",
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        n_shapelet_samples: int = 10000,
        max_shapelets: int | None = None,
        max_shapelet_length: int | None = None,
        estimator=None,
        batch_size: int | None = 100,
        verbose: int = 0,
        transform_limit_in_minutes: int = 0,
        time_limit_in_minutes: int = 0,
        contract_max_n_shapelet_samples: int = np.inf,
        n_jobs: int = 1,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.estimator = estimator
        self.batch_size = batch_size
        self.verbose = verbose
        self.transform_limit_in_minutes = transform_limit_in_minutes
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples
        self.random_state = random_state
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X, y):
        """Fit ShapeletTransformClassifier to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The training data.
        y : array-like, shape = [n_cases]
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
        fit_start = perf_counter() if self.verbose > 0 else None
        X_t = self._fit_stc_shared(X, y)

        estimator_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log(
                f"[STC] Starting estimator fit "
                f"({type(self._estimator).__name__})..."
            )
        self._estimator.fit(X_t, y)
        if self.verbose > 0:
            self._log(
                f"[STC] Finished estimator fit in "
                f"{perf_counter() - estimator_start:.2f}s"
            )
            self._log(f"[STC] Finished fit in {perf_counter() - fit_start:.2f}s")

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_cases]
            Predicted class labels.
        """
        transform_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log("[STC] Starting transform for predict...")
        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, False, -1, -1, -1)
        if self.verbose > 0:
            self._log(
                f"[STC] Finished transform for predict in "
                f"{perf_counter() - transform_start:.2f}s"
            )

        predict_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log("[STC] Starting prediction...")
        pred = self._estimator.predict(X_t)
        if self.verbose > 0:
            self._log(
                f"[STC] Finished prediction in "
                f"{perf_counter() - predict_start:.2f}s"
            )

        return pred

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_cases, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        transform_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log("[STC] Starting transform for predict_proba...")
        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, False, -1, -1, -1)
        if self.verbose > 0:
            self._log(
                f"[STC] Finished transform for predict_proba in "
                f"{perf_counter() - transform_start:.2f}s"
            )

        predict_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log("[STC] Starting probability prediction...")
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            proba = self._estimator.predict_proba(X_t)
        else:
            proba = np.zeros((len(X), self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, len(X)):
                proba[i, np.where(self.classes_ == preds[i])] = 1

        if self.verbose > 0:
            self._log(
                f"[STC] Finished probability prediction in "
                f"{perf_counter() - predict_start:.2f}s"
            )

        return proba

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._fit_predict_proba(X, y)
            ]
        )

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        fit_start = perf_counter() if self.verbose > 0 else None
        X_t = self._fit_stc_shared(X, y)

        estimator_start = perf_counter() if self.verbose > 0 else None
        if (isinstance(self.estimator, RotationForestClassifier)) or (
            self.estimator is None
        ):
            if self.verbose > 0:
                self._log(
                    "[STC] Starting estimator fit and train estimates "
                    "(RotationForest OOB)..."
                )

            proba = self._estimator.fit_predict_proba(X_t, y)
        else:
            if self.verbose > 0:
                self._log(
                    "[STC] Starting estimator fit and train estimates "
                    "(cross-validation)..."
                )

            self._estimator.fit(X_t, y)

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
            if hasattr(estimator, "verbose"):
                estimator.verbose = self.verbose

            proba = cross_val_predict(
                estimator,
                X=X_t,
                y=y,
                cv=cv_size,
                method="predict_proba",
                n_jobs=self._n_jobs,
            )

        if self.verbose > 0:
            self._log(
                f"[STC] Finished estimator fit and train estimates in "
                f"{perf_counter() - estimator_start:.2f}s"
            )
            self._log(f"[STC] Finished fit in {perf_counter() - fit_start:.2f}s")

        return proba

    def _fit_stc_shared(self, X, y):
        self.n_instances_ = len(X)
        self.n_channels_ = X[0].shape[0]
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transform_limit_in_minutes = 0
        self._classifier_limit_in_minutes = 0
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
            batch_size=self.batch_size,
            verbose=self.verbose,
            time_limit_in_minutes=self._transform_limit_in_minutes,
            contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            RotationForestClassifier() if self.estimator is None else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        m = getattr(self._estimator, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes

        if hasattr(self._estimator, "verbose"):
            self._estimator.verbose = self.verbose

        transform_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            if self._transform_limit_in_minutes > 0:
                transform_limit = f"time_limit={self._transform_limit_in_minutes:.2f}m"
            else:
                transform_limit = f"shapelet_samples={self.n_shapelet_samples}"
            self._log(
                f"[STC] Starting fit: n_cases={self.n_instances_}, "
                f"n_channels={self.n_channels_}, {transform_limit}, "
                f"n_jobs={self._n_jobs}"
            )
            self._log("[STC] Starting shapelet transform...")
        X_t = self._transformer.fit_transform(X, y)
        X_t = np.nan_to_num(X_t, False, -1, -1, -1)
        if self.verbose > 0:
            self._log(
                f"[STC] Finished shapelet transform in "
                f"{perf_counter() - transform_start:.2f}s, "
                f"retained={len(self._transformer.shapelets_)}"
            )

        return X_t

    def _log(self, message, level=1):
        """Print a message when the configured verbosity reaches ``level``."""
        if self.verbose >= level:
            print(message, flush=True)  # noqa: T201

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict | list[dict]:
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
