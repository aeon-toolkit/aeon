"""Pipeline with a classifier."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ClassifierPipeline"]


from aeon.base.estimator.compose.collection_pipeline import BaseCollectionPipeline
from aeon.classification.base import BaseClassifier


class ClassifierPipeline(BaseCollectionPipeline, BaseClassifier):
    """Pipeline of transformers and a classifier.

    The `ClassifierPipeline` compositor chains transformers and a single classifier.
    The pipeline is constructed with a list of aeon transformers, plus a classifier,
        i.e., estimators following the BaseTransformer and BaseClassifier interface.
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
    transformers : aeon or sklearn transformer or list of transformers
        A transform or list of transformers to use prior to classification.
        List of tuples (str, transformer) of transformers can also be passed, where
        the str is used to name the transformer.
        The objecst are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.
    classifier : aeon or sklearn classifier
        A classifier to use at the end of the pipeline.
        The object is cloned prior, as such the state of the input will not be modified
        by fitting the pipeline.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of transformers and classifier
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

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
    }

    def __init__(self, transformers, classifier):
        self.classifier = classifier

        super().__init__(transformers=transformers, _estimator=classifier)

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
