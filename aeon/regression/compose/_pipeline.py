"""Pipeline with a regressor."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["RegressorPipeline", "SklearnRegressorPipeline"]

import numpy as np
from deprecated.sphinx import deprecated

from aeon.base import _HeterogenousMetaEstimator
from aeon.base.estimator.compose.collection_pipeline import BaseCollectionPipeline
from aeon.regression.base import BaseRegressor
from aeon.transformations.base import BaseTransformer
from aeon.transformations.compose import TransformerPipeline
from aeon.utils.conversion import convert_collection
from aeon.utils.sklearn import is_sklearn_regressor


class RegressorPipeline(BaseCollectionPipeline, BaseRegressor):
    """Pipeline of transformers and a regressor.

    The `RegressorPipeline` compositor chains transformers and a single regressor.
    The pipeline is constructed with a list of aeon transformers, plus a regressor,
        i.e., estimators following the BaseTransformer amd BaseRegressor interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a regressor `reg`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `reg.fit` with `X` being the output of `trafo[N]`,
        and `y` identical with the input to `self.fit`
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `reg.predict` on the output of `trafoN.transform`,
        and returning the output of `reg.predict`

    Parameters
    ----------
    transformers : aeon or sklearn transformer or list of transformers
        A transform or list of transformers to use prior to regression.
        List of tuples (str, transformer) of transformers can also be passed, where
        the str is used to name the transformer.
        The objecst are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.
    regressor : aeon or sklearn regressor
        A regressor to use at the end of the pipeline.
        The object is cloned prior, as such the state of the input will not be modified
        by fitting the pipeline.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of transformers and regressor
        Clones of transformers and the regressor which are fitted in the pipeline.
        Will always be in (str, estimator) format, even if transformers input is a
        singular transform or list of transformers.

    Examples
    --------
    >>> from aeon.transformations.collection import AutocorrelationFunctionTransformer
    >>> from aeon.datasets import load_covid_3month
    >>> from aeon.regression.compose import RegressorPipeline
    >>> from aeon.regression import DummyRegressor
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> pipeline = RegressorPipeline(
    ...     DummyRegressor(), [AutocorrelationFunctionTransformer(n_lags=10)]
    ... )
    >>> pipeline.fit(X_train, y_train)
    RegressorPipeline(regressor=DummyRegressor(),
                      transformers=[AutocorrelationFunctionTransformer(n_lags=10)])
    >>> y_pred = pipeline.predict(X_test)
    """

    _tags = {
        "X_inner_type": ["numpy3D", "np-list"],  # which type do _fit/_predict accept
    }

    # TODO: remove in v0.9.0
    @deprecated(
        version="0.8.0",
        reason="The position of the regressor and transformers argument for "
        "RegressionPipeline __init__ will be swapped in v0.9.0. Use "
        "keyword arguments to avoid breakage.",
        category=FutureWarning,
    )
    def __init__(self, regressor, transformers):
        self.regressor = regressor

        super().__init__(transformers=transformers, _estimator=regressor)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
        from aeon.transformations.collection import TruncationTransformer
        from aeon.transformations.collection.feature_based import (
            SevenNumberSummaryTransformer,
        )

        return {
            "transformers": [
                TruncationTransformer(truncated_length=5),
                SevenNumberSummaryTransformer(),
            ],
            "regressor": KNeighborsTimeSeriesRegressor(distance="euclidean"),
        }


# TODO: remove in v0.9.0
@deprecated(
    version="0.8.0",
    reason="SklearnRegressorPipeline will be removed in v0.9.0. Use "
    "RegressorPipeline or the sklearn pipeline instead.",
    category=FutureWarning,
)
class SklearnRegressorPipeline(_HeterogenousMetaEstimator, BaseRegressor):
    """Pipeline of transformers and a regressor.

    The `SklearnRegressorPipeline` chains transformers and an single regressor.
        Similar to `RegressorPipeline`, but uses a tabular `sklearn` regressor.
    The pipeline is constructed with a list of aeon transformers, plus a regressor,
        i.e., transformers following the BaseTransformer interface,
        regressor follows the `scikit-learn` regressor interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a regressor `reg`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `reg.fit` with `X` the output of `trafo[N]` converted to numpy,
        and `y` identical with the input to `self.fit`.
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `reg.predict` on the numpy converted output of `trafoN.transform`,
        and returning the output of `reg.predict`.
        Output of `trasfoN.transform` is converted to numpy, as in `fit`.

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `SklearnRegressorPipeline` can also be created by using the magic multiplication
        between `aeon` transformers and `sklearn` regressors,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_reg`
            will result in the same object as  obtained from the constructor
            `SklearnRegressorPipeline(regressor=my_reg, transformers=[t1, t2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    regressor : sklearn regressor, i.e., inheriting from sklearn RegressorMixin
        this is a "blueprint" regressor, state does not change when `fit` is called
    transformers : list of aeon transformers, or
        list of tuples (str, transformer) of aeon transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    regressor_ : sklearn regressor, clone of regressor in `regressor`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of aeon transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from aeon.datasets import load_covid_3month
    >>> from aeon.regression.compose import SklearnRegressorPipeline
    >>> from aeon.transformations.collection.convolution_based import Rocket
    >>> X_train, y_train = load_covid_3month(split="train")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> t1 = Rocket(num_kernels=200)
    >>> pipeline = SklearnRegressorPipeline(KNeighborsRegressor(), [t1])
    >>> pipeline = pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)
    """

    _tags = {
        "X_inner_type": "pd-multiindex",  # which type do _fit/_predict accept
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": True,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    _required_parameters = ["regressor"]

    # no default tag values - these are set dynamically below

    def __init__(self, regressor, transformers):
        from sklearn.base import clone

        self.regressor = regressor
        self.regressor_ = clone(regressor)
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff all transformers can
        # sklearn transformers always support multivariate
        multivariate = not self.transformers_.get_tag("univariate-only", True)
        # can handle missing values iff transformer chain removes missing data
        # sklearn regressors might be able to handle missing data (but no tag there)
        # so better set the tag liberally
        missing = self.transformers_.get_tag("capability:missing_values", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff transformer chain renders series equal length
        # because sklearn regressors require equal length (number of variables) input
        unequal = self.transformers_.get_tag("capability:unequal_length:removes", False)
        # last three tags are always False, since not supported by transformers
        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
            "capability:contractable": False,
            "capability:train_estimate": False,
            "capability:multithreading": False,
        }
        self.set_tags(**tags_to_set)

    @property
    def _transformers(self):
        return self.transformers_._steps

    @_transformers.setter
    def _transformers(self, value):
        self.transformers_._steps = value

    def __rmul__(self, other):
        """Magic * method, return concatenated RegressorPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            # then stick the expanded pipeline in a SklearnRegressorPipeline
            new_pipeline = SklearnRegressorPipeline(
                regressor=self.regressor,
                transformers=trafo_pipeline.steps,
            )
            return new_pipeline
        else:
            return NotImplemented

    def _convert_X_to_sklearn(self, X):
        """Convert X to 2D numpy required by sklearn."""
        Xt = convert_collection(X, "numpy3D")
        return np.reshape(Xt, (Xt.shape[0], Xt.shape[1] * Xt.shape[2]))

    def _fit(self, X, y):
        """Fit time series regressor to training data.

        core logic

        Parameters
        ----------
        X : Training data of type self.get_tag("X_inner_type")
        y : array-like, shape = [n_cases] - the class labels

        Returns
        -------
        self : reference to self.

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        Xt = self.transformers_.fit_transform(X=X, y=y)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        self.regressor_.fit(Xt_sklearn, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        core logic

        Parameters
        ----------
        X : data not used in training, of type self.get_tag("X_inner_type")

        Returns
        -------
        y : predictions of labels for X, np.ndarray
        """
        Xt = self.transformers_.transform(X=X)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        return self.regressor_.predict(Xt_sklearn)

    def get_params(self, deep=True):
        """Get parameters of estimator in `transformers`.

        Parameters
        ----------
        deep : boolean, optional, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "regressor" in kwargs.keys():
            if not is_sklearn_regressor(kwargs["regressor"]):
                raise TypeError('"regressor" arg must be an sklearn regressor')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        regr_keys = self.regressor.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        regr_args = self._subset_dict_keys(
            dict_to_subset=kwargs, keys=regr_keys, prefix="regressor"
        )
        if len(regr_args) > 0:
            self.regressor.set_params(**regr_args)
        if len(trafo_args) > 0:
            self._set_params("_transformers", **trafo_args)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For regressors, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sklearn.neighbors import KNeighborsRegressor

        from aeon.transformations.collection.convolution_based import Rocket

        t1 = Rocket(num_kernels=200, random_state=49)
        c = KNeighborsRegressor()
        return {"transformers": [t1], "regressor": c}
