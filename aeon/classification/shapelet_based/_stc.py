"""A shapelet transform classifier (STC).

Shapelet transform classifier pipeline that simply performs a (configurable) shapelet
transform then builds (by default) a rotation forest classifier on the output.
"""

__maintainer__ = ["TonyBagnall"]
__all__ = ["ShapeletTransformClassifier"]


from time import perf_counter

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state

from aeon.base import CheckpointableMixin
from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn import RotationForestClassifier
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from aeon.utils.validation import check_n_jobs


class ShapeletTransformClassifier(CheckpointableMixin, BaseClassifier):
    """
    A shapelet transform classifier (STC).

    Implementation of the binary shapelet transform classifier pipeline along the lines
    of [1]_, [2]_, but with random shapelet sampling. Transforms the data using the
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
        minimum between ``10 * n_cases_`` and `1000`.
    max_shapelet_length : int or None, default=None
        Lower bound on candidate shapelet lengths for the transform. If ``None``, no
        max length is used
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RotationForestClassifier` classifier is used.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets in the transform.
    verbose : int, default=0
        Level of output printed to the console. ``0`` prints no output, ``1``
        prints STC phase timings and component progress, and ``2`` or greater
        prints detailed progress from the shapelet transform and estimator.
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
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    checkpoint_path : str, pathlib.Path or None, default=None
        Checkpoint file, saved once the shapelet transform completes, at the
        estimator's own safe boundaries when it supports checkpointing, and on
        successful completion. None disables automatic writes. The parent
        directory must exist.
    checkpoint_interval : float or None, default=None
        Minimum minutes between periodic checkpoint writes. None saves only at
        the transform boundary and on successful completion when a path is
        configured. Periodic writes need an estimator that supports
        checkpointing, since the shapelet transform has no safe boundary inside
        it.

    Attributes
    ----------
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_cases_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of channels per case in the training set.
    estimator_ : BaseEstimator
        The fitted base classifier.
    transformer_ : RandomShapeletTransform
        The fitted shapelet transformer.
    fit_elapsed_time_ : float
        Accumulated fitting time in seconds across calls.

    See Also
    --------
    RandomShapeletTransform : The randomly sampled shapelet transform.
    RotationForestClassifier : The default rotation forest classifier used.

    Notes
    -----
    ``resume_fit(X, y)`` continues from saved state using the original training
    data. ``fit`` always starts afresh. Between calls, only
    ``time_limit_in_minutes``, ``n_jobs``, ``verbose`` and the checkpoint
    settings may change; the rest shapes the transform or the estimator, both
    of which a continued fit inherits.

    Fitting has two phases. The shapelet transform has no safe boundary inside
    it, so it is all or nothing: a checkpoint is written once it completes,
    whatever the interval, since it is usually the larger part of the build.
    The estimator fit that follows is resumable only when the estimator itself
    supports checkpointing, as the default ``RotationForestClassifier`` does. It
    then writes through this classifier at its own boundaries, so there is a
    single checkpoint file holding the whole pipeline. With any other estimator,
    a resumed fit rebuilds it from the start on the transformed data, which is
    still the smaller half of the work.

    ``resume_fit`` returns the classifier, not training predictions, so a fit
    begun with ``fit_predict`` or ``fit_predict_proba`` and then resumed gives
    no train estimates.

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
        "capability:checkpointing": True,
        "capability:multithreading": True,
        "capability:unequal_length": True,
        "algorithm_type": "shapelet",
        "X_inner_type": ["np-list", "numpy3D"],
    }

    # the transform is complete before any checkpoint exists, so the parameters
    # shaping it are fixed, as is the estimator a continued fit inherits
    _checkpoint_mutable_params = CheckpointableMixin._checkpoint_mutable_params + (
        "time_limit_in_minutes",
    )

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
        checkpoint_path=None,
        checkpoint_interval=None,
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
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval

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
        fit_start = perf_counter()
        self.fit_elapsed_time_ = 0.0
        self._fit_stc_shared(X, y)
        self._continue_stc(y)
        self.fit_elapsed_time_ = perf_counter() - fit_start
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] "
                f"Finished fit in {self.fit_elapsed_time_:.2f}s"
            )

    def _resume_fit(self, X, y):
        fit_start = perf_counter()
        self._continue_stc(y)
        self.fit_elapsed_time_ += perf_counter() - fit_start
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] "
                f"Finished fit in {self.fit_elapsed_time_:.2f}s total"
            )
        return self

    def _validate_resume_fit(self, X, y):
        check_n_jobs(self.n_jobs)
        if getattr(self, "_transformed_train_data", None) is None:
            raise ValueError(
                "The shapelet transform did not complete, so there is no "
                "transformed data to continue the estimator fit from."
            )

    def _continue_stc(self, y):
        """Fit, or continue fitting, the estimator on the transformed data."""
        self._apply_runtime_settings()
        # a checkpointable estimator that already holds continuation state was
        # interrupted part way through; anything else starts from scratch on
        # the transformed data, which the transform phase already saved
        resumable = isinstance(self.estimator_, CheckpointableMixin) and getattr(
            self.estimator_, "_checkpoint_ready", False
        )
        estimator_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] "
                f"{'Continuing' if resumable else 'Starting'} estimator fit "
                f"({type(self.estimator_).__name__})..."
            )
        if resumable:
            self.estimator_.resume_fit(self._transformed_train_data, y)
        else:
            self.estimator_.fit(self._transformed_train_data, y)
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished estimator fit in "
                f"{perf_counter() - estimator_start:.2f}s"
            )

    def _split_time_limits(self):
        """Divide any overall contract between the transform and estimator."""
        transform_limit = 0
        classifier_limit = 0
        if self.time_limit_in_minutes > 0:
            # contracting 2/3 transform (with 1/5 of that taken away for final
            # transform), 1/3 classifier
            third = self.time_limit_in_minutes / 3
            classifier_limit = third
            transform_limit = (third * 2) / 5 * 4
        elif self.transform_limit_in_minutes > 0:
            transform_limit = self.transform_limit_in_minutes
        return transform_limit, classifier_limit

    def _apply_runtime_settings(self):
        """Push the settings that may change between calls onto the estimator."""
        self._n_jobs = check_n_jobs(self.n_jobs)
        _, self._classifier_limit_in_minutes = self._split_time_limits()

        m = getattr(self.estimator_, "n_jobs", None)
        if m is not None:
            self.estimator_.n_jobs = self._n_jobs

        m = getattr(self.estimator_, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self.estimator_.time_limit_in_minutes = self._classifier_limit_in_minutes

        # only pass verbosity to RotationForestClassifier, other estimators such as
        # scikit-learn forests interpret verbose levels differently
        if isinstance(self.estimator_, RotationForestClassifier):
            self.estimator_.verbose = self.verbose

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
            self._log(f"[{type(self).__name__}] Starting transform for predict...")
        X_t = self.transformer_.transform(X)
        X_t = np.nan_to_num(X_t, False, -1, -1, -1)
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished transform for predict in "
                f"{perf_counter() - transform_start:.2f}s"
            )

        predict_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log(f"[{type(self).__name__}] Starting prediction...")
        pred = self.estimator_.predict(X_t)
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished prediction in "
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
            self._log(
                f"[{type(self).__name__}] Starting transform for predict_proba..."
            )
        X_t = self.transformer_.transform(X)
        X_t = np.nan_to_num(X_t, False, -1, -1, -1)
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished transform for predict_proba in "
                f"{perf_counter() - transform_start:.2f}s"
            )

        predict_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            self._log(f"[{type(self).__name__}] Starting probability prediction...")
        m = getattr(self.estimator_, "predict_proba", None)
        if callable(m):
            proba = self.estimator_.predict_proba(X_t)
        else:
            proba = np.zeros((len(X), self.n_classes_))
            preds = self.estimator_.predict(X_t)
            for i in range(0, len(X)):
                proba[i, np.where(self.classes_ == preds[i])] = 1

        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished probability prediction in "
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
                    f"[{type(self).__name__}] "
                    "Starting estimator fit and train estimates "
                    "(RotationForest OOB)..."
                )

            proba = self.estimator_.fit_predict_proba(X_t, y)
        else:
            if self.verbose > 0:
                self._log(
                    f"[{type(self).__name__}] "
                    "Starting estimator fit and train estimates "
                    "(cross-validation)..."
                )

            self.estimator_.fit(X_t, y)

            m = getattr(self.estimator_, "predict_proba", None)
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
                f"[{type(self).__name__}] "
                "Finished estimator fit and train estimates in "
                f"{perf_counter() - estimator_start:.2f}s"
            )
            self._log(
                f"[{type(self).__name__}] "
                f"Finished fit in {perf_counter() - fit_start:.2f}s"
            )

        return proba

    def _fit_stc_shared(self, X, y):
        self.n_instances_ = len(X)
        self.n_channels_ = X[0].shape[0]
        self._n_jobs = check_n_jobs(self.n_jobs)

        (
            self._transform_limit_in_minutes,
            self._classifier_limit_in_minutes,
        ) = self._split_time_limits()

        self.transformer_ = RandomShapeletTransform(
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

        self.estimator_ = _clone_estimator(
            RotationForestClassifier() if self.estimator is None else self.estimator,
            self.random_state,
        )
        self._apply_runtime_settings()

        # a checkpointable estimator holds no path of its own: its boundaries
        # save this classifier, which holds it and the transform it needs
        if isinstance(self.estimator_, CheckpointableMixin):
            self.estimator_._set_checkpoint_parent(self)

        transform_start = perf_counter() if self.verbose > 0 else None
        if self.verbose > 0:
            if self._transform_limit_in_minutes > 0:
                transform_limit = f"time_limit={self._transform_limit_in_minutes:.2f}m"
            else:
                transform_limit = f"shapelet_samples={self.n_shapelet_samples}"
            self._log(
                f"[{type(self).__name__}] Starting fit: n_cases={self.n_instances_}, "
                f"n_channels={self.n_channels_}, {transform_limit}, "
                f"n_jobs={self._n_jobs}"
            )
            self._log(f"[{type(self).__name__}] Starting shapelet transform...")
        X_t = self.transformer_.fit_transform(X, y)
        X_t = np.nan_to_num(X_t, False, -1, -1, -1)
        if self.verbose > 0:
            self._log(
                f"[{type(self).__name__}] Finished shapelet transform in "
                f"{perf_counter() - transform_start:.2f}s, "
                f"retained={len(self.transformer_.shapelets)}"
            )

        # the transform has no safe boundary inside it, so this is the first
        # point a checkpoint can be taken. It is always written when a path is
        # configured, interval or not, because it is usually the larger part of
        # the build and repeating it is the cost a checkpoint exists to avoid
        self._transformed_train_data = X_t
        self._checkpoint_ready = True
        self._checkpoint_if_due(force=True)

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
        elif parameter_set == "checkpointing":
            # the forest is the only resumable phase, so it must run for more
            # than one batch of trees
            return {
                "estimator": RotationForestClassifier(n_estimators=3),
                "n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
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
