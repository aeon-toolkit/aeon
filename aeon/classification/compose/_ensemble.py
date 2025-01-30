"""Configurable time series classification ensemble."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ClassifierEnsemble"]


import numpy as np
from sklearn.utils import check_random_state

from aeon.base._estimators.compose.collection_ensemble import BaseCollectionEnsemble
from aeon.classification.base import BaseClassifier
from aeon.classification.sklearn._wrapper import SklearnClassifierWrapper
from aeon.utils.sklearn import is_sklearn_classifier


class ClassifierEnsemble(BaseCollectionEnsemble, BaseClassifier):
    """Weighted ensemble of classifiers with fittable ensemble weight.

    Parameters
    ----------
    classifiers : list of aeon and/or sklearn classifiers or list of tuples
        Estimators to be used in the ensemble.
        A list of tuples (str, estimator) can also be passed, where the str is used to
        name the estimator.
        The objects are cloned prior. As such, the state of the input will not be
        modified by fitting the ensemble.
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
    majority_vote : bool, default=False
        If True, the ensemble predictions are the class with the majority of class
        votes from the ensemble.
        If False, the ensemble predictions are the class with the highest probability
        summed from ensemble members.
    random_state : int, RandomState instance or None, default=None
        Random state used to fit the estimators and break ties. If None, no random
        state is set for ensemble members (but they may still be seeded prior to
        input) and tie breaking.
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    ensemble_ : list of tuples (str, estimator) of estimators
        Clones of estimators in classifiers which are fitted in the ensemble.
        Will always be in (str, estimator) format regardless of classifiers input.
    weights_ : dict
        Weights of estimators using the str names as keys.

    See Also
    --------
    RegressorEnsemble : An ensemble for regression tasks.
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        classifiers,
        weights=None,
        cv=None,
        metric=None,
        metric_probas=False,
        majority_vote=False,
        random_state=None,
    ):
        self.classifiers = classifiers
        self.majority_vote = majority_vote

        wclf = [self._wrap_sklearn(clf) for clf in self.classifiers]

        super().__init__(
            _ensemble=wclf,
            weights=weights,
            cv=cv,
            metric=metric,
            metric_probas=metric_probas,
            random_state=random_state,
            _ensemble_input_name="classifiers",
        )

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X."""
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
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
        dists = np.zeros((len(X), self.n_classes_))

        if self.majority_vote:
            # Call predict on each classifier, add the weighted predictions to the
            # current probabilities
            for clf_name, clf in self.ensemble_:
                preds = clf.predict(X=X)
                for i in range(X.shape[0]):
                    dists[i, self._class_dictionary[preds[i]]] += self.weights_[
                        clf_name
                    ]
        else:
            # Call predict_proba on each classifier, multiply the probabilities by the
            # classifiers weight then add them to the current probabilities
            for clf_name, clf in self.ensemble_:
                proba = clf.predict_proba(X=X)
                dists += proba * self.weights_[clf_name]

        # Make each instances probability array sum to 1 and return
        y_proba = dists / dists.sum(axis=1, keepdims=True)

        return y_proba

    @staticmethod
    def _wrap_sklearn(clf):
        if isinstance(clf, tuple):
            if is_sklearn_classifier(clf[1]):
                return clf[0], SklearnClassifierWrapper(clf[1])
            else:
                return clf
        elif is_sklearn_classifier(clf):
            return SklearnClassifierWrapper(clf)
        else:
            return clf

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from aeon.classification import DummyClassifier
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

        return {
            "classifiers": [
                KNeighborsTimeSeriesClassifier._create_test_instance(),
                DummyClassifier._create_test_instance(),
            ],
            "weights": [2, 1],
        }
