"""Base class for pipelines in series collection modules.

i.e. classification, regression and clustering.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseCollectionPipeline"]

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from aeon.base import (
    BaseAeonEstimator,
    BaseCollectionEstimator,
    ComposableEstimatorMixin,
)
from aeon.base._base import _clone_estimator


class BaseCollectionPipeline(ComposableEstimatorMixin, BaseCollectionEstimator):
    """Base class for composable pipelines in collection based modules.

    Parameters
    ----------
    transformers : aeon or sklearn transformer or list of transformers
        A transform or list of transformers to use prior to fitting or predicting.
        List of tuples (str, transformer) of transformers can also be passed, where
        the str is used to name the transformer.
        The objects are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.
    _estimator : aeon or sklearn estimator
        A estimator to use at the end of the pipeline.
        The object is cloned prior, as such the state of the input will not be modified
        by fitting the pipeline.
    random_state : int, RandomState instance or None, default=None
        Random state used to fit the estimators. If None, no random state is set for
        pipeline components (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of transformers and estimator
        Clones of transformers and the estimator which are fitted in the pipeline.
        Will always be in (str, estimator) format, even if transformers input is a
        singular transform or list of transformers.

    See Also
    --------
    ClassifierPipeline : A pipeline for classification tasks.
    RegressorPipeline : A pipeline for regression tasks.
    """

    # Attribute name containing an iterable of processed (str, estimator) tuples
    # with unfitted estimators and unique names. Used in get_params and set_params
    _estimators_attr = "_steps"
    # Attribute name containing an iterable of fitted (str, estimator) tuples.
    # Used in get_fitted_params
    _fitted_estimators_attr = "steps_"

    @abstractmethod
    def __init__(self, transformers, _estimator, random_state=None):
        self.transformers = transformers
        self._estimator = _estimator
        self.random_state = random_state

        self._steps = (
            [t for t in transformers]
            if isinstance(transformers, list)
            else [transformers]
        )
        if _estimator is not None:
            self._steps.append(_estimator)

        self._check_estimators(
            self._steps,
            attr_name="_steps",
            class_type=BaseEstimator,
        )
        self._steps = self._convert_estimators(self._steps, clone_estimators=False)

        super().__init__()

        # can handle multivariate if: both estimator and all transformers can,
        #   *or* transformer chain removes multivariate
        multivariate_tags = [
            (
                e[1].get_tag(
                    "capability:multivariate",
                    raise_error=False,
                    tag_value_default=False,
                )
                if isinstance(e[1], BaseAeonEstimator)
                else False
            )
            for e in self._steps
        ]

        multivariate_rm_tag = False
        for e in self._steps:
            if (
                isinstance(e[1], BaseAeonEstimator)
                and e[1].get_tag(
                    "capability:multivariate",
                    raise_error=False,
                    tag_value_default=False,
                )
                and e[1].get_tag("output_data_type", raise_error=False) == "Tabular"
            ):
                multivariate_rm_tag = True
                break
            elif not isinstance(e[1], BaseAeonEstimator) or not e[1].get_tag(
                "capability:multivariate", raise_error=False, tag_value_default=False
            ):
                break

        multivariate = all(multivariate_tags) or multivariate_rm_tag

        # can handle missing values if: both estimator and all transformers can,
        #   *or* transformer chain removes missing data
        missing_tags = [
            (
                e[1].get_tag(
                    "capability:missing_values",
                    raise_error=False,
                    tag_value_default=False,
                )
                if isinstance(e[1], BaseAeonEstimator)
                else False
            )
            for e in self._steps
        ]

        missing_rm_tag = False
        for e in self._steps:
            if (
                isinstance(e[1], BaseAeonEstimator)
                and e[1].get_tag(
                    "capability:missing_values",
                    raise_error=False,
                    tag_value_default=False,
                )
                and e[1].get_tag(
                    "removes_missing_values", raise_error=False, tag_value_default=False
                )
            ):
                missing_rm_tag = True
                break
            elif not isinstance(e[1], BaseAeonEstimator) or not e[1].get_tag(
                "capability:missing_values", raise_error=False, tag_value_default=False
            ):
                break

        missing = all(missing_tags) or missing_rm_tag

        # can handle unequal length if: estimator can and transformers can,
        #   *or* transformer chain renders the series equal length
        #   *or* transformer chain transforms the series to a tabular format
        unequal_tags = [
            (
                e[1].get_tag(
                    "capability:unequal_length",
                    raise_error=False,
                    tag_value_default=False,
                )
                if isinstance(e[1], BaseAeonEstimator)
                else False
            )
            for e in self._steps
        ]

        unequal_rm_tag = False
        for e in self._steps:
            if (
                isinstance(e[1], BaseAeonEstimator)
                and e[1].get_tag(
                    "capability:unequal_length",
                    raise_error=False,
                    tag_value_default=False,
                )
                and (
                    e[1].get_tag(
                        "removes_unequal_length",
                        raise_error=False,
                        tag_value_default=False,
                    )
                    or e[1].get_tag("output_data_type", raise_error=False) == "Tabular"
                )
            ):
                unequal_rm_tag = True
                break
            elif not isinstance(e[1], BaseAeonEstimator) or not e[1].get_tag(
                "capability:unequal_length", raise_error=False, tag_value_default=False
            ):
                break

        unequal = all(unequal_tags) or unequal_rm_tag

        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
        }
        self.set_tags(**tags_to_set)

    def _fit(self, X, y):
        """Fit time series estimator to training data.

        Parameters
        ----------
        X : Training data of type self.get_tag("X_inner_type")
        y : array-like, shape = [n_cases] - the target values

        Returns
        -------
        self : reference to self.
        """
        self._clone_steps()

        # fit transforms sequentially
        Xt = X
        for i in range(len(self.steps_) - 1):
            Xt = self.steps_[i][1].fit_transform(X=Xt, y=y)
        # fit estimator
        self.steps_[-1][1].fit(X=Xt, y=y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        Parameters
        ----------
        X : data not used in training, of type self.get_tag("X_inner_type")

        Returns
        -------
        y : predictions of target values for X, np.ndarray
        """
        # transform
        Xt = X
        for i in range(len(self.steps_) - 1):
            Xt = self.steps_[i][1].transform(X=Xt)
        # predict
        return self.steps_[-1][1].predict(X=Xt)

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : data to predict y with, of type self.get_tag("X_inner_type")

        Returns
        -------
        y : predictions of probabilities for target values of X, np.ndarray
        """
        # transform
        Xt = X
        for i in range(len(self.steps_) - 1):
            Xt = self.steps_[i][1].transform(X=Xt)
        # predict
        return self.steps_[-1][1].predict_proba(X=Xt)

    def _fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform sequences in X.

        Parameters
        ----------
        X : data of type self.get_tag("X_inner_type")

        Returns
        -------
        Xt : transformed data
        """
        self._clone_steps()

        # transform
        Xt = X
        for i in range(len(self.steps_)):
            Xt = self.steps_[i][1].fit_transform(X=Xt)
        return Xt

    def _transform(self, X, y=None) -> np.ndarray:
        """Transform sequences in X.

        Parameters
        ----------
        X : data of type self.get_tag("X_inner_type")

        Returns
        -------
        Xt : transformed data
        """
        # transform
        Xt = X
        for i in range(len(self.steps_)):
            Xt = self.steps_[i][1].transform(X=Xt)
        return Xt

    def _clone_steps(self):
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            self.steps_ = [
                (
                    step[0],
                    _clone_estimator(
                        step[1], random_state=rng.randint(np.iinfo(np.int32).max)
                    ),
                )
                for step in self._steps
            ]
        else:
            self.steps_ = [(step[0], _clone_estimator(step[1])) for step in self._steps]
