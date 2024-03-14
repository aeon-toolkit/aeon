"""Base class for pipelines in collection data based modules.

i.e. classification, regression and clustering.
"""

__maintainer__ = ["MatthewMiddlehurst"]

import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator

from aeon.base import BaseCollectionEstimator, BaseEstimator, _HeterogenousMetaEstimator


class BaseCollectionPipeline(_HeterogenousMetaEstimator, BaseCollectionEstimator):
    """Base class for composable pipelines in collection based modules.

    Parameters
    ----------
    transformers : aeon or sklearn transformer or list of transformers
        A transform or list of transformers to use prior to fitting or predicting.
        List of tuples (str, transformer) of transformers can also be passed, where
        the str is used to name the transformer.
        The objecst are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.
    _estimator : aeon or sklearn estimator
        A estimator to use at the end of the pipeline.
        The object is cloned prior, as such the state of the input will not be modified
        by fitting the pipeline.

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

    def __init__(self, transformers, _estimator):
        self.transformers = transformers
        self._estimator = _estimator

        self._steps = (
            [t for t in transformers]
            if isinstance(transformers, list)
            else [transformers]
        )
        self._steps.append(_estimator)
        self._steps = self._check_estimators(
            self._steps,
            attr_name="_steps",
            cls_type=SklearnBaseEstimator,
            clone_ests=False,
        )

        super().__init__()

        # can handle multivariate if: both estimator and all transformers can
        multivariate_tags = [
            (
                e[1].get_tag("capability:multivariate", False, raise_error=False)
                if isinstance(e[1], BaseEstimator)
                else False
            )
            for e in self._steps
        ]

        multivariate_rm_tag = False
        for e in self._steps:
            if (
                isinstance(e[1], BaseEstimator)
                and e[1].get_tag("capability:multivariate", False, raise_error=False)
                and e[1].get_tag("output_data_type", raise_error=False) == "Tabular"
            ):
                multivariate_rm_tag = True
                break
            elif not isinstance(e[1], BaseEstimator) or not e[1].get_tag(
                "capability:multivariate", False, raise_error=False
            ):
                break

        multivariate = all(multivariate_tags) or multivariate_rm_tag

        # can handle missing values if: both estimator and all transformers can,
        #   *or* transformer chain removes missing data
        missing_tags = [
            (
                e[1].get_tag("capability:missing_values", False, raise_error=False)
                if isinstance(e[1], BaseEstimator)
                else False
            )
            for e in self._steps
        ]

        missing_rm_tag = False
        for e in self._steps:
            if (
                isinstance(e[1], BaseEstimator)
                and e[1].get_tag("capability:missing_values", False, raise_error=False)
                and e[1].get_tag(
                    "capability:missing_values:removes", False, raise_error=False
                )
            ):
                missing_rm_tag = True
                break
            elif not isinstance(e[1], BaseEstimator) or not e[1].get_tag(
                "capability:missing_values", False, raise_error=False
            ):
                break

        missing = all(missing_tags) or missing_rm_tag

        # can handle unequal length if: estimator can and transformers can,
        #   *or* transformer chain renders the series equal length
        #   *or* transformer chain transforms the series to a tabular format
        unequal_tags = [
            (
                e[1].get_tag("capability:unequal_length", False, raise_error=False)
                if isinstance(e[1], BaseEstimator)
                else False
            )
            for e in self._steps
        ]

        unequal_rm_tag = False
        for e in self._steps:
            if (
                isinstance(e[1], BaseEstimator)
                and e[1].get_tag("capability:unequal_length", False, raise_error=False)
                and (
                    e[1].get_tag(
                        "capability:unequal_length:removes", False, raise_error=False
                    )
                    or e[1].get_tag("output_data_type", raise_error=False) == "Tabular"
                )
            ):
                unequal_rm_tag = True
                break
            elif not isinstance(e[1], BaseEstimator) or not e[1].get_tag(
                "capability:unequal_length", False, raise_error=False
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

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        self.steps_ = self._check_estimators(
            self._steps, attr_name="steps_", cls_type=SklearnBaseEstimator
        )

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
