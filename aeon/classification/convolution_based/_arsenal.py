"""Arsenal classifier.

kernel based ensemble of ROCKET classifiers.
"""

__maintainer__ = []
__all__ = ["Arsenal"]

import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class Arsenal(BaseClassifier):
    """
    Arsenal ensemble.

    Overview: an ensemble of ROCKET transformers using RidgeClassifierCV base
    classifier. Weights each classifier using the accuracy from the ridge
    cross-validation. Allows for generation of probability estimates at the
    expense of scalability compared to RocketClassifier.

    Parameters
    ----------
    num_kernels : int, default=2,000
        Number of kernels for each ROCKET transform.
    n_estimators : int, default=25
        Number of estimators to build for the ensemble.
    rocket_transform : str, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket","minirocket","multirocket"].
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=100
        Max number of estimators when time_limit_in_minutes is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_cases_ : int
        The number of train cases.
    n_channels_ : int
        The number of dimensions per case.
    n_timepoints_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    weights_ : list of shape (n_estimators) of float
        Weight of each estimator in the ensemble.
    n_estimators_ : int
        The number of estimators in the ensemble.

    See Also
    --------
    RocketClassifier
        Arsenal is an ensemble of RocketClassifier.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/classifiers/kernel_based/Arsenal.java>`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." arXiv preprint arXiv:2104.07551 (2021).

    Examples
    --------
    >>> from aeon.classification.convolution_based import Arsenal
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test =load_unit_test(split="test")
    >>> clf = Arsenal(num_kernels=100, n_estimators=5)
    >>> clf.fit(X_train, y_train)
    Arsenal(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        num_kernels=2000,
        n_estimators=25,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=100,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.n_estimators = n_estimators
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.n_cases_ = 0
        self.n_channels_ = 0
        self.n_timepoints_ = 0
        self.estimators_ = []
        self.weights_ = []

        self._weight_sum = 0

        super().__init__()

    def _fit(self, X, y):
        """Fit Arsenal to training data.

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
        ending in "_" and sets is_fitted flag to True.
        """
        self._fit_arsenal(X, y)

        return self

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
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._predict_proba(X)
            ]
        )

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
        y_probas = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.estimators_[i],
                i,
            )
            for i in range(self.n_estimators_)
        )

        return np.around(
            np.sum(y_probas, axis=0) / (np.ones(self.n_classes_) * self._weight_sum), 8
        )

    def _fit_predict(self, X, y) -> np.ndarray:
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self._fit_predict_proba(X, y)
            ]
        )

    def _fit_predict_proba(self, X, y) -> np.ndarray:
        Xt = self._fit_arsenal(X, y, keep_transformed_data=True)

        rng = check_random_state(self.random_state)

        p = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(self._train_probas_for_estimator)(
                Xt,
                y,
                i,
                check_random_state(rng.randint(np.iinfo(np.int32).max)),
            )
            for i in range(self.n_estimators_)
        )
        y_probas, weights, oobs = zip(*p)

        results = np.sum(y_probas, axis=0)
        divisors = np.zeros(self.n_cases_)
        for n, oob in enumerate(oobs):
            for inst in oob:
                divisors[inst] += weights[n]

        for i in range(self.n_cases_):
            results[i] = (
                np.ones(self.n_classes_) * (1 / self.n_classes_)
                if divisors[i] == 0
                else results[i] / (np.ones(self.n_classes_) * divisors[i])
            )

        return results

    def _fit_arsenal(self, X, y, keep_transformed_data=False):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if self.rocket_transform == "rocket":
            base_rocket = Rocket(num_kernels=self.num_kernels)
        elif self.rocket_transform == "minirocket":
            if self.n_channels_ > 1:
                base_rocket = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                )
            else:
                base_rocket = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                )
        elif self.rocket_transform == "multirocket":
            if self.n_channels_ > 1:
                base_rocket = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                )
            else:
                base_rocket = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        if time_limit > 0:
            self.n_estimators_ = 0
            self.estimators_ = []
            Xt = []

            while (
                train_time < time_limit
                and self.n_estimators_ < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                    delayed(self._fit_ensemble_estimator)(
                        _clone_estimator(
                            base_rocket,
                            (
                                None
                                if self.random_state is None
                                else (
                                    255 if self.random_state == 0 else self.random_state
                                )
                                * 37
                                * (i + 1)
                            ),
                        ),
                        X,
                        y,
                        keep_transformed_data=keep_transformed_data,
                    )
                    for i in range(self._n_jobs)
                )

                estimators, transformed_data = zip(*fit)

                self.estimators_ += estimators
                Xt += transformed_data

                self.n_estimators_ += self._n_jobs
                train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(self._fit_ensemble_estimator)(
                    _clone_estimator(
                        base_rocket,
                        (
                            None
                            if self.random_state is None
                            else (255 if self.random_state == 0 else self.random_state)
                            * 37
                            * (i + 1)
                        ),
                    ),
                    X,
                    y,
                    keep_transformed_data=keep_transformed_data,
                )
                for i in range(self.n_estimators)
            )

            self.estimators_, Xt = zip(*fit)
            self.n_estimators_ = self.n_estimators

        self.weights_ = []
        self._weight_sum = 0
        for rocket_pipeline in self.estimators_:
            weight = rocket_pipeline.steps[2][1].best_score_
            self.weights_.append(weight)
            self._weight_sum += weight

        return Xt

    def _fit_ensemble_estimator(self, rocket, X, y, keep_transformed_data):
        transformed_x = rocket.fit_transform(X)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(transformed_x, y)
        ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        ridge.fit(scaler.transform(transformed_x), y)
        return [
            make_pipeline(rocket, scaler, ridge),
            transformed_x if keep_transformed_data else None,
        ]

    def _predict_proba_for_estimator(self, X, classifier, idx):
        preds = classifier.predict(X)
        weights = np.zeros((X.shape[0], self.n_classes_))
        for i in range(X.shape[0]):
            weights[i, self._class_dictionary[preds[i]]] += self.weights_[idx]
        return weights

    def _train_probas_for_estimator(self, Xt, y, idx, rng):
        indices = range(self.n_cases_)
        subsample = rng.choice(self.n_cases_, size=self.n_cases_)
        oob = [n for n in indices if n not in subsample]

        results = np.zeros((self.n_cases_, self.n_classes_))
        if not oob:
            return results, 1, oob

        clf = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        clf.fit(Xt[idx][subsample], y[subsample])
        preds = clf.predict(Xt[idx][oob])

        weight = clf.steps[1][1].best_score_

        for n, pred in enumerate(preds):
            results[oob[n]][self._class_dictionary[pred]] += weight

        return results, weight, oob

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Arsenal provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates
                "contracting" - used in classifiers that set the
                    "capability:contractable" tag to True to test contacting
                    functionality

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        if parameter_set == "results_comparison":
            return {"num_kernels": 20, "n_estimators": 5}
        elif parameter_set == "contracting":
            return {
                "time_limit_in_minutes": 5,
                "num_kernels": 10,
                "contract_max_n_estimators": 2,
            }
        else:
            return {"num_kernels": 10, "n_estimators": 2}
