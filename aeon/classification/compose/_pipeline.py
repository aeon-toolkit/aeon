"""Pipeline with a classifier."""

__maintainer__ = []
__all__ = ["ClassifierPipeline"]


import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator

from aeon.base import BaseEstimator, _HeterogenousMetaEstimator
from aeon.classification.base import BaseClassifier


class ClassifierPipeline(_HeterogenousMetaEstimator, BaseClassifier):
    """Pipeline of transformers and a classifier.

    The `ClassifierPipeline` compositor chains transformers and a single classifier.
    The pipeline is constructed with a list of aeon transformers, plus a classifier,
        i.e., estimators following the BaseTransformer resp BaseClassifier interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a classifier `clf`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes state by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `clf.fit` with `X` being the output of `trafo[N]`,
        and `y` identical with the input to `self.fit`
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clf.predict` on the output of `trafoN.transform`,
        and returning the output of `clf.predict`
    `predict_proba(X)` - result is of executing `trafo1.transform`, `trafo2.transform`,
        etc, with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clf.predict_proba` on the output of `trafoN.transform`,
        and returning the output of `clf.predict_proba`

    Parameters
    ----------
    classifier : aeon or sklearn classifier
        A classifier to use at the end of the pipeline.
        The object is cloned prior, as such the state of the input will not be modified
        by fitting the pipeline.
    transformers : aeon or sklearn transformer or list of transformers
        A transform or list of transformers to use prior to classification.
        List of tuples (str, transformer) of transformers can also be passed, where
        the str is used to name the transformer.
        The objecst are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of tansformers and classifier
        Clones of transformers and the classifier which are fitted in the pipeline.
        Will always be in (str, estimator) format, even if transformers input is a
        singular transform or list of transformers.

    Examples
    --------
    >>> from aeon.transformations.collection.interpolate import TSInterpolator
    >>> from aeon.classification.convolution_based import RocketClassifier
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.compose import ClassifierPipeline
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> pipeline = ClassifierPipeline(
    ...     TSInterpolator(length=10), RocketClassifier(num_kernels=50)
    ... )
    >>> pipeline.fit(X_train, y_train)
    ClassifierPipeline(...)
    >>> y_pred = pipeline.predict(X_test)
    """

    def __init__(self, transformers, classifier):
        self.classifier = classifier
        self.transformers = transformers

        self._steps = (
            [t for t in transformers]
            if isinstance(transformers, list)
            else [transformers]
        )
        self._steps.append(classifier)
        self._steps = self._check_estimators(
            self._steps,
            attr_name="_steps",
            cls_type=SklearnBaseEstimator,
            clone_ests=False,
        )

        super().__init__()

        # can handle multivariate if: both classifier and all transformers can
        multivariate_tags = [
            (
                t[1].get_tag("capability:multivariate", False, raise_error=False)
                if isinstance(t[1], BaseEstimator)
                else False
            )
            for t in self._steps
        ]
        multivariate = all(multivariate_tags)

        # can handle missing values if: both classifier and all transformers can,
        #   *or* transformer chain removes missing data
        missing_tags = [
            (
                t[1].get_tag("capability:missing_values", False, raise_error=False)
                if isinstance(t[1], BaseEstimator)
                else False
            )
            for t in self._steps
        ]
        missing_rm_rags = [
            (
                t[1].get_tag(
                    "capability:missing_values:removes", False, raise_error=False
                )
                if isinstance(t[1], BaseEstimator)
                else False
            )
            for t in self._steps
        ]
        missing = all(missing_tags) or any(missing_rm_rags)

        # can handle unequal length if: classifier can and transformers can,
        #   *or* transformer chain renders the series equal length
        unequal_tags = [
            (
                t[1].get_tag("capability:unequal_length", False, raise_error=False)
                if isinstance(t[1], BaseEstimator)
                else False
            )
            for t in self._steps
        ]
        unequal_rm_tags = [
            (
                t[1].get_tag(
                    "capability:unequal_length:removes", False, raise_error=False
                )
                if isinstance(t[1], BaseEstimator)
                else False
            )
            for t in self._steps
        ]
        unequal = all(unequal_tags) or any(unequal_rm_tags)

        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
        }
        self.set_tags(**tags_to_set)

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : Training data of type self.get_tag("X_inner_type")
        y : array-like, shape = [n_instances] - the class labels

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
        # fit classifier
        self.steps_[-1][1].fit(X=Xt, y=y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict labels for sequences in X.

        Parameters
        ----------
        X : data not used in training, of type self.get_tag("X_inner_type")

        Returns
        -------
        y : predictions of labels for X, np.ndarray
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
        y : predictions of probabilities for class values of X, np.ndarray
        """
        # transform
        Xt = X
        for i in range(len(self.steps_) - 1):
            Xt = self.steps_[i][1].transform(X=Xt)
        # predict
        return self.steps_[-1][1].predict_proba(X=Xt)

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
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        from aeon.transformations.collection import TruncationTransformer
        from aeon.transformations.collection.feature_based import (
            SevenNumberSummaryTransformer,
        )

        return {
            "transformers": [
                TruncationTransformer(truncated_length=5),
                SevenNumberSummaryTransformer(),
            ],
            "classifier": KNeighborsTimeSeriesClassifier(distance="euclidean"),
        }
