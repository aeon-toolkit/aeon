"""Pipeline with a classifier."""

__author__ = ["fkiraly", "MatthewMiddlehurst", "TonyBagnall"]
__all__ = ["ClassifierPipeline", "SklearnClassifierPipeline"]


import numpy as np
from deprecated.sphinx import deprecated
from sklearn.base import BaseEstimator as SklearnBaseEstimator

from aeon.base import BaseEstimator, _HeterogenousMetaEstimator
from aeon.classification.base import BaseClassifier
from aeon.transformations.base import BaseTransformer
from aeon.transformations.compose import TransformerPipeline
from aeon.utils.conversion import convert_collection
from aeon.utils.sklearn import is_sklearn_classifier


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
    ...     RocketClassifier(num_kernels=50), TSInterpolator(length=10)
    ... )
    >>> pipeline.fit(X_train, y_train)
    ClassifierPipeline(...)
    >>> y_pred = pipeline.predict(X_test)
    """

    # TODO: remove in v0.8.0
    @deprecated(
        version="0.7.0",
        reason="The position of the classifier and transformers argument for "
        "ClassifierPipeline __init__ will be swapped in v0.8.0. Use "
        "keyword arguments to avoid breakage.",
        category=FutureWarning,
    )
    def __init__(self, classifier, transformers):
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

    # TODO: remove in v0.8.0
    @deprecated(
        version="0.7.0",
        reason="The ClassifierPipeline __rmul__ (*) functionality will be removed "
        "in v0.8.0.",
        category=FutureWarning,
    )
    def __rmul__(self, other):
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            return ClassifierPipeline(
                classifier=self.classifier,
                transformers=trafo_pipeline.steps,
            )
        else:
            return NotImplemented

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


# TODO: remove in v0.8.0
@deprecated(
    version="0.7.0",
    reason="SklearnClassifierPipeline will be removed in v0.8.0. Use "
    "ClassifierPipeline or the sklearn pipeline instead.",
    category=FutureWarning,
)
class SklearnClassifierPipeline(_HeterogenousMetaEstimator, BaseClassifier):
    """Pipeline of transformers and a classifier.

    The `SklearnClassifierPipeline` chains transformers and a single classifier.
        Similar to `ClassifierPipeline`, but uses a tabular `sklearn` classifier.
    The pipeline is constructed with a list of aeon transformers, plus a classifier,
        i.e., transformers following the BaseTransformer interface,
        classifier follows the `scikit-learn` classifier interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN` and a classifier `clf`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes styte by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `clf.fit` with `X` the output of `trafo[N]` converted to numpy,
        and `y` identical with the input to `self.fit`.
    `predict(X)` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clf.predict` on the numpy converted output of `trafoN.transform`,
        and returning the output of `clf.predict`.
        Output of `trasfoN.transform` is converted to numpy, as in `fit`.
    `predict_proba(X)` - result is of executing `trafo1.transform`, `trafo2.transform`,
        etc, with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `clf.predict_proba` on the output of `trafoN.transform`,
        and returning the output of `clf.predict_proba`.
        Output of `trasfoN.transform` is converted to numpy, as in `fit`.

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `SklearnClassifierPipeline` can also be created by using the magic multiplication
        between `aeon` transformers and `sklearn` classifiers,
            and `my_trafo1`, `my_trafo2` inherit from `BaseTransformer`, then,
            for instance, `my_trafo1 * my_trafo2 * my_clf`
            will result in the same object as  obtained from the constructor
            `SklearnClassifierPipeline(classifier=my_clf, transformers=[t1, t2])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a transformer

    Parameters
    ----------
    classifier : sklearn classifier, i.e., inheriting from sklearn ClassifierMixin
        this is a "blueprint" classifier, state does not change when `fit` is called
    transformers : list of aeon transformers, or
        list of tuples (str, transformer) of aeon transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    classifier_ : sklearn classifier, clone of classifier in `classifier`
        this clone is fitted in the pipeline when `fit` is called
    transformers_ : list of tuples (str, transformer) of aeon transformers
        clones of transformers in `transformers` which are fitted in the pipeline
        is always in (str, transformer) format, even if transformers is just a list
        strings not passed in transformers are unique generated strings
        i-th transformer in `transformers_` is clone of i-th in `transformers`

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from aeon.transformations.exponent import ExponentTransformer
    >>> from aeon.transformations.summarize import SummaryTransformer
    >>> from aeon.datasets import load_unit_test
    >>> from aeon.classification.compose import SklearnClassifierPipeline
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> t1 = ExponentTransformer()
    >>> t2 = SummaryTransformer()
    >>> pipeline = SklearnClassifierPipeline(KNeighborsClassifier(), [t1, t2])
    >>> pipeline = pipeline.fit(X_train, y_train)
    >>> y_pred = pipeline.predict(X_test)
    """

    # no default tag values - these are set dynamically below

    def __init__(self, classifier, transformers):
        from sklearn.base import clone

        self.classifier = classifier
        self.classifier_ = clone(classifier)
        self.transformers = transformers
        self.transformers_ = TransformerPipeline(transformers)

        super().__init__()

        # can handle multivariate iff all transformers can
        # sklearn transformers always support multivariate
        multivariate = not self.transformers_.get_tag("univariate-only", True)
        # can handle missing values iff transformer chain removes missing data
        # sklearn classifiers might be able to handle missing data (but no tag there)
        # so better set the tag liberally
        missing = self.transformers_.get_tag("capability:missing_values", False)
        missing = missing or self.transformers_.get_tag(
            "capability:missing_values:removes", False
        )
        # can handle unequal length iff transformer chain renders series equal length
        # because sklearn classifiers require equal length (number of variables) input
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
        """Magic * method, return concatenated ClassifierPipeline, transformers on left.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        ClassifierPipeline object, concatenation of `other` (first) with `self` (last).
        """
        if isinstance(other, BaseTransformer):
            # use the transformers dunder to get a TransformerPipeline
            trafo_pipeline = other * self.transformers_
            return SklearnClassifierPipeline(
                classifier=self.classifier,
                transformers=trafo_pipeline.steps,
            )
        else:
            return NotImplemented

    def _convert_X_to_sklearn(self, X):
        """Convert X to 2D numpy required by sklearn."""
        Xt = convert_collection(X, "numpy3D")
        return np.reshape(Xt, (Xt.shape[0], Xt.shape[1] * Xt.shape[2]))

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        core logic

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
        Xt = self.transformers_.fit_transform(X=X, y=y)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        self.classifier_.fit(Xt_sklearn, y)

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
        return self.classifier_.predict(Xt_sklearn)

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
        Xt = self.transformers_.transform(X)
        if not hasattr(self.classifier_, "predict_proba"):
            # if sklearn classifier does not have predict_proba
            return BaseClassifier._predict_proba(self, X)
        Xt_sklearn = self._convert_X_to_sklearn(Xt)
        return self.classifier_.predict_proba(Xt_sklearn)

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
        params = {}
        trafo_params = self._get_params("_transformers", deep=deep)
        params.update(trafo_params)

        return params

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        if "classifier" in kwargs and not is_sklearn_classifier(kwargs["classifier"]):
            raise TypeError('"classifier" arg must be an sklearn classifier')
        trafo_keys = self._get_params("_transformers", deep=True).keys()
        classif_keys = self.classifier.get_params(deep=True).keys()
        trafo_args = self._subset_dict_keys(dict_to_subset=kwargs, keys=trafo_keys)
        classif_args = self._subset_dict_keys(
            dict_to_subset=kwargs, keys=classif_keys, prefix="classifier"
        )
        if len(classif_args) > 0:
            self.classifier.set_params(**classif_args)
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

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
        """
        from sklearn.neighbors import KNeighborsClassifier

        from aeon.transformations.collection.convolution_based import Rocket

        t1 = Rocket(num_kernels=200, random_state=49)
        c = KNeighborsClassifier()
        return {"transformers": [t1], "classifier": c}
