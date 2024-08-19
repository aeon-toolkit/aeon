"""Base class for collection ensembles."""

import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state

from aeon.base import BaseCollectionEstimator, BaseEstimator, _HeterogenousMetaEstimator
from aeon.base._base import _clone_estimator


class BaseCollectionEnsemble(_HeterogenousMetaEstimator, BaseCollectionEstimator):
    """Weighted ensemble of collection estimators with fittable ensemble weight.

    Parameters
    ----------
    _estimators : list of aeon and/or sklearn estimators or list of tuples
        Estimators to be used in the ensemble. The str is used to name the estimator.
        List of tuples (str, estimator) of estimators can also be passed, where
        the str is used to name the estimator.
        The objects are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.
    weights : float, or iterable of float, default=None
        If float, ensemble weight for estimator i will be train score to this power.
        If iterable of float, must be equal length as _estimators. Ensemble weight for
            _estimator i will be weights[i].
        If None, all estimators have equal weight.
    cv : None, int, or sklearn cross-validation object, default=None
        Only used if weights is a float. The method used to generate a performance
        estimation from the training data set i.e. cross-validation.
        If None, predictions are made using that estimators fit_predict or
            fit_predict_proba methods. These are somtimes overridden for efficient
            performance evaluations, i.e. out-of-bag predictions.
        If int or sklearn object input, the parameter is passed directly to the cv
            parameter of the cross_val_predict function from sklearn.
    metric : sklearn performance metric function, default=accuracy_score
        Only used if weights is a float. The metric used to evaluate the estimators.
    metric_probas : bool, default=False
        Only used if weights is a float. Whether to generate predictions via predict
        (False) or probabilities via predict_proba (True) for use in the metric.
    random_state : int, RandomState instance or None, default=None
        Random state used to fit the estimators. If None, no random state is set for
        ensemble members (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    ensemble_ : list of tuples (str, estimator) of estimators
        Clones of estimators in _estimators which are fitted in the ensemble.
        Will always be in (str, estimator) format regardless of _estimators input.
    weights_ : dict
        Weights of estimators using the str names as keys.

    See Also
    --------
    ClassifierEnsemble : A pipeline for classification tasks.
    RegressorEnsemble : A pipeline for regression tasks.
    """

    def __init__(
        self,
        _estimators,
        weights=None,
        cv=None,
        metric=None,
        metric_probas=False,
        random_state=None,
    ):
        self._estimators = _estimators
        self.weights = weights
        self.cv = cv
        self.metric = metric
        self.metric_probas = metric_probas
        self.random_state = random_state

        self.ensemble_ = self._check_estimators(
            self._estimators,
            attr_name="_estimators",
            cls_type=SklearnBaseEstimator,
            clone_ests=False,
        )

        super().__init__()

        # can handle multivariate if all estimators can
        multivariate = all(
            [
                (
                    e[1].get_tag("capability:multivariate", False, raise_error=False)
                    if isinstance(e[1], BaseEstimator)
                    else False
                )
                for e in self.ensemble_
            ]
        )

        # can handle missing values if all estimators can
        missing = all(
            [
                (
                    e[1].get_tag("capability:missing_values", False, raise_error=False)
                    if isinstance(e[1], BaseEstimator)
                    else False
                )
                for e in self.ensemble_
            ]
        )

        # can handle unequal length if all estimators can
        unequal = all(
            [
                (
                    e[1].get_tag("capability:unequal_length", False, raise_error=False)
                    if isinstance(e[1], BaseEstimator)
                    else False
                )
                for e in self.ensemble_
            ]
        )

        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
        }
        self.set_tags(**tags_to_set)

    def _fit(self, X, y):
        self._clone_steps()

        msg = (
            "weights must be a float, dict, or iterable of floats of length equal "
            "to _estimators"
        )
        if self.weights is None:
            self.weights_ = {x[0]: 1 for x in self.ensemble_}
        elif isinstance(self.weights, (float, int)):
            self.weights_ = {}
        else:
            try:
                if len(self.weights) != len(self.ensemble_):
                    raise ValueError(msg)
                self.weights_ = {
                    x[0]: self.weights[i] for i, x in enumerate(self.ensemble_)
                }
            except TypeError:
                raise ValueError(msg)

        if not isinstance(self.weights, (float, int)):
            for _, estimator in self.ensemble_:
                estimator.fit(X=X, y=y)
        # if weights are calculated by training loss, we fit_predict and evaluate
        else:
            if self.metric is None:
                metric = accuracy_score if is_classifier(self) else mean_squared_error
            else:
                metric = self.metric

            for name, estimator in self.ensemble_:
                # estimate predictions from train data
                if self.cv is None:
                    preds = (
                        estimator.fit_predict_proba(X=X, y=y)
                        if self.metric_probas
                        else estimator.fit_predict(X=X, y=y)
                    )
                else:
                    preds = cross_val_predict(
                        estimator,
                        X=X,
                        y=y,
                        cv=self.cv,
                        method="predict_proba" if self.metric_probas else "predict",
                    )
                    # train final model
                    estimator.fit(X, y)

                self.weights_[name] = metric(y, preds) ** self.weights

        return self

    def _clone_steps(self):
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            self.ensemble_ = [
                (
                    step[0],
                    _clone_estimator(
                        step[1], random_state=rng.randint(np.iinfo(np.int32).max)
                    ),
                )
                for step in self.ensemble_
            ]
        else:
            self.ensemble_ = [
                (step[0], _clone_estimator(step[1])) for step in self.ensemble_
            ]
