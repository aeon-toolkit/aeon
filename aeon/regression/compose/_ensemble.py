"""Configurable time series regression ensemble."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RegressorEnsemble"]


import numpy as np

from aeon.base._estimators.compose.collection_ensemble import BaseCollectionEnsemble
from aeon.regression import BaseRegressor
from aeon.regression.sklearn._wrapper import SklearnRegressorWrapper
from aeon.utils.sklearn import is_sklearn_regressor


class RegressorEnsemble(BaseCollectionEnsemble, BaseRegressor):
    """Weighted ensemble of regressors with fittable ensemble weight.

    Parameters
    ----------
    regressors : list of aeon and/or sklearn regressors or list of tuples
        Estimators to be used in the ensemble.
        A list of tuples (str, estimator) can also be passed, where the str is used to
        name the estimator.
        The objects are cloned prior. As such, the state of the input will not be
        modified by fitting the ensemble.
    weights : float, or iterable of float, default=None
        If float, ensemble weight for estimator i will be train score to this power.
        If iterable of float, must be equal length as _estimators. Ensemble weight for
            _estimator i will be weights[i]. A dict containing members of _estimators
            and weights is also acceptable.
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
    random_state : int, RandomState instance or None, default=None
        Random state used to fit the estimators. If None, no random state is set for
        ensemble members (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    ensemble_ : list of tuples (str, estimator) of estimators
        Clones of estimators in regressors which are fitted in the ensemble.
        Will always be in (str, estimator) format regardless of regressors input.
    weights_ : dict
        Weights of estimators using the str names as keys.

    See Also
    --------
    ClassifierEnsemble : An ensemble for classification tasks.
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(
        self,
        regressors,
        weights=None,
        cv=None,
        metric=None,
        random_state=None,
    ):
        self.regressors = regressors

        wreg = [self._wrap_sklearn(clf) for clf in self.regressors]

        super().__init__(
            _ensemble=wreg,
            weights=weights,
            cv=cv,
            metric=metric,
            metric_probas=False,
            random_state=random_state,
            _ensemble_input_name="regressors",
        )

    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X."""
        preds = np.zeros(len(X))

        for reg_name, reg in self.ensemble_:
            preds += reg.predict(X=X) * self.weights_[reg_name]

        return preds / np.sum(list(self.weights_.values()))

    @staticmethod
    def _wrap_sklearn(reg):
        if isinstance(reg, tuple):
            if is_sklearn_regressor(reg[1]):
                return reg[0], SklearnRegressorWrapper(reg[1])
            else:
                return reg
        elif is_sklearn_regressor(reg):
            return SklearnRegressorWrapper(reg)
        else:
            return reg

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
        from aeon.regression import DummyRegressor
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

        return {
            "regressors": [
                KNeighborsTimeSeriesRegressor._create_test_instance(),
                DummyRegressor._create_test_instance(),
            ],
            "weights": [2, 1],
        }
