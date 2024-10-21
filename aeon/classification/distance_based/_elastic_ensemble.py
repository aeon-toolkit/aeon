"""The Elastic Ensemble (EE).

An ensemble of elastic nearest neighbour classifiers.
"""

__maintainer__ = []
__all__ = ["ElasticEnsemble"]

import math
import time
from itertools import product
from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_predict,
)

from aeon.classification.base import BaseClassifier
from aeon.classification.distance_based._time_series_neighbors import (
    KNeighborsTimeSeriesClassifier,
)
from aeon.utils.numba.general import slope_derivative_2d


class ElasticEnsemble(BaseClassifier):
    """The Elastic Ensemble (EE) of time series distance measures.

    The Elastic Ensemble [1]_ is an ensemble of 1-NN classifiers using elastic
    distances (as defined in aeon.distances). By default, each 1-NN classifier
    is tuned over 100 parameter values and the ensemble vote is weighted by
    an estimate of accuracy formed on the train set.

    Parameters
    ----------
    distance_measures : str or list of str, default="all"
      A list of strings identifying which distance measures to include. Valid values
      are one or more of: ``euclidean``, ``dtw``, ``wdtw``, ``ddtw``, ``wddtw``,
      ``lcss``, ``erp``, ``msm``, ``twe``. The default value ``all`` means that all
      the previously listed distances are used.
    proportion_of_param_options : float, default=1
      The proportion of the parameter grid space to search optional.
    proportion_train_in_param_finding : float, default=1
      The proportion of the train set to use in the parameter search optional.
    proportion_train_for_test : float, default=1
      The proportion of the train set to use in classifying new cases optional.
    n_jobs : int, default=1
      The number of jobs to run in parallel for both `fit` and `predict`.
      ``-1`` means using all processors.
    random_state : int, default=0
        If `int`, random_state is the seed used by the random number generator;
    verbose : int, default=0
      If ``>0``, then prints out debug information.
    majority_vote: boolean, default = False
      Whether to use majority vote or weighted vote.

    Attributes
    ----------
    estimators_ : list
      A list storing all classifiers.
    train_accs_by_classifier_ : np.ndarray
      Store the train accuracies of the classifiers.
    constituent_build_times_ : array of float
        build time for each member of the ensemble.

    References
    ----------
    .. [1] Jason Lines and Anthony Bagnall, "Time Series Classification with Ensembles
        of Elastic Distance Measures", Data Mining and Knowledge Discovery, 29(3), 2015.
        https://link.springer.com/article/10.1007/s10618-014-0361-2

    Examples
    --------
    >>> from aeon.classification.distance_based import ElasticEnsemble
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = ElasticEnsemble(
    ...     proportion_of_param_options=0.1,
    ...     proportion_train_for_test=0.1,
    ...     distance_measures = ["dtw","ddtw"],
    ...     majority_vote=True,
    ... )
    >>> clf.fit(X_train, y_train)
    ElasticEnsemble(...)
    >>> y_pred = clf.predict(X_test)
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
        distance_measures: Union[str, list[str]] = "all",
        proportion_of_param_options: float = 1.0,
        proportion_train_in_param_finding: float = 1.0,
        proportion_train_for_test: float = 1.0,
        n_jobs: int = 1,
        random_state: int = 0,
        verbose: int = 0,
        majority_vote: bool = False,
    ) -> None:
        self.distance_measures = distance_measures
        self.proportion_train_in_param_finding = proportion_train_in_param_finding
        self.proportion_of_param_options = proportion_of_param_options
        self.proportion_train_for_test = proportion_train_for_test
        self.n_jobs = n_jobs
        self.majority_vote = majority_vote
        self.random_state = random_state
        self.verbose = verbose
        self.estimators_ = None
        self.train_accs_by_classifier_ = None
        self.constituent_build_times_ = None

        super().__init__()

    def _fit(self, X, y):
        """Build an ensemble of 1-NN classifiers from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            or list of [n_cases] np.ndarray shape (n_channels, n_timepoints_i)
            The training input samples.

        y : array-like, shape = (n_cases,) The class labels.

        Returns
        -------
        self : object
        """
        if self.distance_measures == "all":
            self._distance_measures = [
                "dtw",
                "ddtw",
                "wdtw",
                "wddtw",
                "lcss",
                "erp",
                "msm",
                "euclidean",
                "twe",
            ]
        else:
            self._distance_measures = self.distance_measures

        # Derivative DTW (DDTW) uses the regular DTW algorithm on data that
        # are transformed into derivatives.
        if self._distance_measures.__contains__(
            "ddtw"
        ) or self._distance_measures.__contains__("wddtw"):
            der_X = []  # use list to allow for unequal length
            for x in X:
                der_X.append(slope_derivative_2d(x))
            if isinstance(X, np.ndarray):
                der_X = np.array(der_X)
        else:
            der_X = None

        self.train_accs_by_classifier_ = np.zeros(len(self._distance_measures))
        self.estimators_ = [None] * len(self._distance_measures)
        rand = np.random.RandomState(self.random_state)
        # The default EE uses all training instances for setting parameters,
        # and 100 parameter options per elastic measure. The
        # prop_train_in_param_finding and prop_of_param_options attributes of this class
        # can be used to control this however, using fewer cases to optimise
        # parameters on the training data and/or using less parameter options.
        #
        # For using fewer training instances the appropriate number of cases must be
        # sampled from the data. This is achieved through the use of a deterministic
        # StratifiedShuffleSplit
        #
        # For using fewer parameter options a RandomizedSearchCV is used in
        # place of a GridSearchCV

        param_train_x = None
        der_param_train_x = None
        param_train_y = None

        # If using less cases for parameter optimisation, use the scikit
        # StratifiedShuffleSplit:
        if self.proportion_train_in_param_finding < 1:
            if self.verbose > 0:
                print(  # noqa: T201
                    "Restricting training cases for parameter optimisation: ", end=""
                )
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 - self.proportion_train_in_param_finding,
                random_state=rand,
            )
            for train_index, _ in sss.split(X, y):
                param_train_x = X[train_index, :]
                param_train_y = y[train_index]
                if der_X is not None:
                    der_param_train_x = der_X[train_index, :]
                if self.verbose > 0:
                    print(  # noqa: T201
                        f"using{len(param_train_x)} training cases instead of "
                        f"{len(X)} for parameter optimisation"
                    )
        # else, use the full training data for optimising parameters
        else:
            if self.verbose > 0:
                print(  # noqa: T201
                    "Using all training cases for parameter optimisation"
                )
            param_train_x = X
            param_train_y = y
            if der_X is not None:
                der_param_train_x = der_X

        self.constituent_build_times_ = []

        if self.verbose > 0:
            print(  # noqa: T201
                f"Using{(100 * self.proportion_of_param_options)} parameter options"
            )
        for dm in range(0, len(self._distance_measures)):
            this_measure = self._distance_measures[dm]

            # uses the appropriate training data as required (either full or
            # smaller sample as per the StratifiedShuffleSplit)
            param_train_to_use = param_train_x
            full_train_to_use = X
            if this_measure == "ddtw" or this_measure == "wddtw":
                param_train_to_use = der_param_train_x
                full_train_to_use = der_X
                if this_measure == "ddtw":
                    this_measure = "dtw"
                elif this_measure == "wddtw":
                    this_measure = "wdtw"

            start_build_time = time.time()
            if self.verbose > 0:
                if (
                    self._distance_measures[dm] == "ddtw"
                    or self._distance_measures[dm] == "wddtw"
                ):
                    print(  # noqa: T201
                        f"Currently evaluating {self._distance_measures[dm]} "
                        f"implemented as {this_measure} with pre-transformed "
                        f"derivative data)"
                    )
                else:
                    print(  # noqa: T201
                        f"Currently evaluating {self._distance_measures[dm]}"
                    )

            # If 100 parameter options are being considered per measure,
            # use a GridSearchCV
            if self.proportion_of_param_options == 1:
                grid = GridSearchCV(
                    estimator=KNeighborsTimeSeriesClassifier(
                        distance=this_measure, n_neighbors=1
                    ),
                    param_grid=ElasticEnsemble._get_100_param_options(
                        self._distance_measures[dm], X
                    ),
                    cv=LeaveOneOut(),
                    scoring="accuracy",
                    n_jobs=self._n_jobs,
                    verbose=self.verbose,
                )
                grid.fit(param_train_to_use, param_train_y)

            # Else, used RandomizedSearchCV to randomly sample parameter
            # options for each measure
            else:
                grid = RandomizedSearchCV(
                    estimator=KNeighborsTimeSeriesClassifier(
                        distance=this_measure, n_neighbors=1
                    ),
                    param_distributions=ElasticEnsemble._get_100_param_options(
                        self._distance_measures[dm], X
                    ),
                    n_iter=math.ceil(100 * self.proportion_of_param_options),
                    cv=LeaveOneOut(),
                    scoring="accuracy",
                    n_jobs=self._n_jobs,
                    random_state=rand,
                    verbose=self.verbose,
                )
                grid.fit(param_train_to_use, param_train_y)

            if self.majority_vote:
                acc = 1
            # once the best parameter option has been estimated on the
            # training data, perform a final pass with this parameter option
            # to get the individual predictions with cross_cal_predict (
            # Note: optimisation potentially possible here if a GridSearchCV
            # was used previously. TO-DO: determine how to extract
            # predictions for the best param option from GridSearchCV)
            else:
                best_model = KNeighborsTimeSeriesClassifier(
                    n_neighbors=1,
                    distance=this_measure,
                    distance_params=grid.best_params_["distance_params"],
                    n_jobs=self._n_jobs,
                )
                preds = cross_val_predict(
                    best_model, full_train_to_use, y, cv=LeaveOneOut()
                )
                acc = accuracy_score(y, preds)

            if self.verbose > 0:
                print(  # noqa: T201
                    f"Training acc for {self._distance_measures[dm]}: {acc}"
                )

            # Finally, reset the classifier for this measure and parameter
            # option, ready to be called for test classification
            best_model = KNeighborsTimeSeriesClassifier(
                n_neighbors=1,
                distance=this_measure,
                distance_params=grid.best_params_["distance_params"],
            )
            best_model.fit(full_train_to_use, y)
            end_build_time = time.time()

            self.constituent_build_times_.append(str(end_build_time - start_build_time))
            self.estimators_[dm] = best_model
            self.train_accs_by_classifier_[dm] = acc
        return self

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            or list of [n_cases] numpy arrays size n_channels, n_timepoints_i)

        Returns
        -------
        y : array-like, shape = (n_cases, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        if self._distance_measures.__contains__(
            "ddtw"
        ) or self._distance_measures.__contains__("wddtw"):
            der_X = []  # use list to allow for unequal length
            for x in X:
                der_X.append(slope_derivative_2d(x))
            if isinstance(X, np.ndarray):
                der_X = np.array(der_X)
        else:
            der_X = None

        output_probas = []
        train_sum = 0

        for c in range(0, len(self.estimators_)):
            if (
                self._distance_measures[c] == "ddtw"
                or self._distance_measures[c] == "wddtw"
            ):
                test_X_to_use = der_X
            else:
                test_X_to_use = X
            this_train_acc = self.train_accs_by_classifier_[c]
            this_probas = np.multiply(
                self.estimators_[c].predict_proba(test_X_to_use), this_train_acc
            )
            output_probas.append(this_probas)
            train_sum += this_train_acc

        output_probas = np.sum(output_probas, axis=0)
        output_probas = np.divide(output_probas, train_sum)
        return output_probas

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            or list of [n_cases] np.ndarray shape (n_channels, n_timepoints_i)
            The training input samples.

        Returns
        -------
        y : array-like, shape = (n_cases) The class labels.
            Predicted class labels.
        """
        probas = self._predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds

    def get_metric_params(self) -> dict:
        """Return the parameters for the distance metrics used.

        Returns
        -------
        params : dict
            The distance measures and the list of their parameter values.
        """
        return {
            self._distance_measures[dm]: str(self.estimators_[dm]._distance_params)
            for dm in range(len(self.estimators_))
        }

    @staticmethod
    def _get_100_param_options(distance_measure: str, train_x=None):
        """Generate 100 parameter values for each classifier.

        Parameters
        ----------
        distance_measure : str, the name of the distance measure.

        train_x : np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            or list of [n_cases] np.ndarray shape (n_channels, n_timepoints_i)
            The training input samples.
        """

        def get_inclusive(min_val: float, max_val: float, num_vals: float):
            inc = (max_val - min_val) / (num_vals - 1)
            return np.arange(min_val, max_val + inc / 2, inc)

        if distance_measure == "dtw" or distance_measure == "ddtw":
            return {"distance_params": [{"window": x / 100} for x in range(0, 100)]}
        elif distance_measure == "wdtw" or distance_measure == "wddtw":
            return {"distance_params": [{"g": x / 100} for x in range(0, 100)]}
        elif distance_measure == "lcss":
            if train_x is None:
                raise ValueError(f"Error need train data for {distance_measure}")
            train_std = np.std(train_x)
            epsilons = get_inclusive(train_std * 0.2, train_std, 10)
            deltas = get_inclusive(0, 0.25, 10)
            a = list(product(epsilons, deltas))
            return {
                "distance_params": [
                    {"epsilon": a[x][0], "window": a[x][1]} for x in range(0, len(a))
                ]
            }
        elif distance_measure == "erp":
            if train_x is None:
                raise ValueError(f"Error need train data for {distance_measure}")
            train_std = np.std(train_x)
            band_sizes = get_inclusive(0, 0.25, 10)
            g_vals = get_inclusive(train_std * 0.2, train_std, 10)
            b_and_g = list(product(band_sizes, g_vals))
            return {
                "distance_params": [
                    {"window": b_and_g[x][0], "g": b_and_g[x][1]}
                    for x in range(0, len(b_and_g))
                ]
            }
        elif distance_measure == "msm":
            a = get_inclusive(0.01, 0.1, 25)
            b = get_inclusive(0.1, 1, 26)
            c = get_inclusive(1, 10, 26)
            d = get_inclusive(10, 100, 26)
            return {
                "distance_params": [
                    {"c": x} for x in np.concatenate([a, b[1:], c[1:], d[1:]])
                ]
            }
        elif distance_measure == "euclidean":
            return {"distance_params": [{}]}  # No parameters for Euclidean distance
        elif distance_measure == "twe":
            band_sizes = get_inclusive(0, 0.25, 5)
            nu_values = get_inclusive(0.0001, 0.001, 10)
            lmbda_values = get_inclusive(1.0, 2.0, 2)
            b_and_nu_and_lmbda = list(product(band_sizes, nu_values, lmbda_values))
            return {
                "distance_params": [
                    {
                        "window": b_and_nu_and_lmbda[x][0],
                        "nu": b_and_nu_and_lmbda[x][1],
                        "lmbda": b_and_nu_and_lmbda[x][2],
                    }
                    for x in range(0, len(b_and_nu_and_lmbda))
                ]
            }

        else:
            raise NotImplementedError(
                "EE does not currently support: " + str(distance_measure)
            )

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
        if parameter_set == "results_comparison":
            return {
                "proportion_of_param_options": 0.1,
                "proportion_train_for_test": 0.1,
                "majority_vote": True,
                "distance_measures": ["dtw", "ddtw", "wdtw"],
            }
        else:
            return {
                "proportion_of_param_options": 0.01,
                "proportion_train_for_test": 0.1,
                "majority_vote": True,
                "distance_measures": ["dtw"],
            }
