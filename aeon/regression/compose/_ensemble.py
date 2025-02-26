"""Configurable time series regression ensemble."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RegressorEnsemble"]

import numpy as np

from aeon.base._estimators.compose.collection_ensemble import BaseCollectionEnsemble
from aeon.regression import BaseRegressor
from aeon.regression.sklearn._wrapper import SklearnRegressorWrapper
from aeon.utils.sklearn import is_sklearn_regressor


class RegressorEnsemble(BaseCollectionEnsemble, BaseRegressor):
    """Weighted ensemble of regressors with configurable ensemble weight.

    Parameters
    ----------
    regressors : list of ``aeon`` and/or ``sklearn`` regressors or list of tuples
        Estimators to be used in the ensemble.
        A list of tuples (``str``, estimator) can also be passed, where the ``str`` is used to
        name the estimator.
        The objects are cloned prior to training. As such, the state of the input will not be
        modified by fitting the ensemble.
    weights : ``float``, iterable of ``float``, or ``dict``, default=``None``
        If ``float``, ensemble weight for estimator ``i`` will be train score to this power.
        If iterable of ``float``, must be of equal length as ``regressors``. The weight for
        each estimator will be taken from the corresponding position.
        If ``dict``, must map estimator names (as defined in ``ensemble_``) to their weights.
        If ``None``, all estimators have equal weight.
    cv : ``None``, ``int``, or ``sklearn`` cross-validation object, default=``None``
        Only used if ``weights`` is a ``float``. Specifies the cross-validation strategy
        to estimate model performance.
        If ``None``, predictions are obtained using each estimator's ``fit_predict`` method.
        If ``int`` or ``sklearn`` object, it is passed directly to ``cross_val_predict`` from ``sklearn``.
    metric : ``sklearn`` performance metric function, default=``None``
        Only used if ``weights`` is a ``float``. The metric used to evaluate the estimators.
    random_state : ``int``, ``RandomState`` instance, or ``None``, default=``None``
        Random state used to fit the estimators. If ``None``, no random state is set for
        ensemble members (but they may still be seeded prior to input).
        If ``int``, ``random_state`` is the seed used by the random number generator.
        If ``RandomState`` instance, ``random_state`` is the random number generator.

    Attributes
    ----------
    ensemble_ : list of tuples (``str``, estimator)
        Cloned estimators fitted in the ensemble. Always in (``str``, estimator) format,
        regardless of the input format in ``regressors``.
    weights_ : ``dict``
        Dictionary mapping estimator names to their assigned weights.

    See Also
    --------
    ``ClassifierEnsemble`` : An ensemble for classification tasks.
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
        """Predicts labels for sequences in ``X``."""
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
        parameter_set : ``str``, default=``"default"``
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : ``dict`` or list of ``dict``, default=``{}``
            Parameters to create testing instances of the class.
            Each ``dict`` are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test instance.
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
