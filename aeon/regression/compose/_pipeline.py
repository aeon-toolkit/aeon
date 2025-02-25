"""Pipeline with a regressor."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RegressorPipeline"]

from aeon.base._estimators.compose.collection_pipeline import BaseCollectionPipeline
from aeon.regression.base import BaseRegressor


class RegressorPipeline(BaseCollectionPipeline, BaseRegressor):
    """
    Pipeline of transformers and a regressor.

    The ``RegressorPipeline`` compositor chains transformers and a single regressor.
    The pipeline is constructed with a list of ``aeon`` transformers and a regressor,
    i.e., estimators following the ``BaseTransformer`` and ``BaseRegressor`` interface.
    
    The transformer list can be either unnamed (a simple list of transformers) or named
    (a list of ``(str, estimator)`` pairs).

    For a list of transformers ``trafo1``, ``trafo2``, ..., ``trafoN`` and a regressor ``reg``,
    the pipeline behaves as follows:
    
    - ``fit(X, y)``: Runs ``trafo1.fit_transform(X)``, then sequentially applies
      ``trafo2.fit_transform()`` to the output of ``trafo1``, and so on.
      Finally, ``reg.fit()`` is run with ``X`` being the output of ``trafoN.fit_transform()``,
      while ``y`` remains unchanged.
    
    - ``predict(X)``: Sequentially applies ``trafo1.transform()``, then ``trafo2.transform()``, etc.
      The final transformed ``X`` is passed to ``reg.predict()``, and the output is returned.

    Parameters
    ----------
    transformers : ``aeon`` or ``sklearn`` transformer or list of transformers
        A transformer or list of transformers to use prior to regression.
        A list of tuples ``(str, transformer)`` can also be passed, where
        the ``str`` is used to name the transformer.
        The objects are cloned before fitting, ensuring that their state is not modified.
    regressor : ``aeon`` or ``sklearn`` regressor
        The regressor to use at the end of the pipeline.
        The object is cloned before fitting to maintain an unmodified state.
    random_state : ``int``, ``RandomState`` instance, or ``None``, default=``None``
        Random state used to fit the estimators.
        - If ``None``, no specific random state is set.
        - If ``int``, it is used as a random seed.
        - If ``RandomState`` instance, it acts as the random number generator.

    Attributes
    ----------
    steps_ : list of tuples ``(str, estimator)`` of transformers and regressor
        Clones of transformers and the regressor which are fitted in the pipeline.
        Always in the ``(str, estimator)`` format, even if the transformers input is
        a singular transformer or a list of transformers.

    Examples
    --------
    >>> from aeon.transformations.collection import AutocorrelationFunctionTransformer
    >>> from aeon.datasets import load_covid_3month
    >>> from aeon.regression.compose import RegressorPipeline
    >>> from aeon.regression import DummyRegressor
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> pipeline = RegressorPipeline(
    ...     [AutocorrelationFunctionTransformer(n_lags=10)], DummyRegressor(),
    ... )
    >>> pipeline.fit(X_train, y_train)
    RegressorPipeline(regressor=DummyRegressor(),
                      transformers=[AutocorrelationFunctionTransformer(n_lags=10)])
    >>> y_pred = pipeline.predict(X_test)
   """


    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],
    }

    def __init__(self, transformers, regressor, random_state=None):
        self.regressor = regressor

        super().__init__(
            transformers=transformers, _estimator=regressor, random_state=random_state
        )

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
            Each ``dict`` contains parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test instance.
        """
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
        from aeon.transformations.collection import Truncator
        from aeon.transformations.collection.feature_based import SevenNumberSummary

        return {
            "transformers": [
                Truncator(truncated_length=5),
                SevenNumberSummary(),
            ],
            "regressor": KNeighborsTimeSeriesRegressor(distance="euclidean"),
        }
