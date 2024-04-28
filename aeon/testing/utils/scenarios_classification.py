"""Test scenarios for classification and regression.

Contains TestScenario concrete children to run in tests for classifiers/regressors.
"""

__maintainer__ = []

__all__ = [
    "scenarios_classification",
    "scenarios_early_classification",
    "scenarios_regression",
]

from inspect import isclass

from aeon.base import BaseObject
from aeon.classification.base import BaseClassifier
from aeon.classification.early_classification import BaseEarlyClassifier
from aeon.regression.base import BaseRegressor
from aeon.testing.utils.data_gen import (
    make_example_3d_numpy,
    make_example_unequal_length,
)
from aeon.testing.utils.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


class ClassifierTestScenario(TestScenario, BaseObject):
    """Generic test scenario for classifiers."""

    def get_args(self, key, obj=None, deepcopy_args=True):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        # use same args for predict-like functions as for predict
        if key in ["predict_proba", "decision_function"]:
            key = "predict"

        return super().get_args(key=key, obj=obj, deepcopy_args=deepcopy_args)

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """

        def get_tag(obj, tag_name):
            if isclass(obj):
                return obj.get_class_tag(tag_name)
            else:
                return obj.get_tag(tag_name)

        regr_or_classf = (BaseClassifier, BaseEarlyClassifier, BaseRegressor)

        # applicable only if obj inherits from BaseClassifier, BaseEarlyClassifier or
        #   BaseRegressor. currently we test both classifiers and regressors using these
        #   scenarios
        if not isinstance(obj, regr_or_classf) and not issubclass(obj, regr_or_classf):
            return False

        # if X is multivariate, applicable only if can handle multivariate
        is_multivariate = not self.get_tag("X_univariate")
        if is_multivariate and not get_tag(obj, "capability:multivariate"):
            return False

        # if X is unequal length, applicable only if can handle unequal length
        is_unequal_length = self.get_tag("X_unequal_length")
        if is_unequal_length and not get_tag(obj, "capability:unequal_length"):
            return False

        return True


X, y = make_example_3d_numpy(n_cases=10, n_timepoints=20, random_state=RAND_SEED)
X_test, _ = make_example_3d_numpy(n_cases=5, n_timepoints=20, random_state=RAND_SEED)

X_mv, _ = make_example_3d_numpy(
    n_cases=10, n_channels=2, n_timepoints=20, random_state=RAND_SEED
)
X_test_mv, _ = make_example_3d_numpy(
    n_cases=5, n_channels=2, n_timepoints=20, random_state=RAND_SEED
)

X_ul, _ = make_example_unequal_length(
    n_cases=10, max_n_timepoints=15, min_n_timepoints=10, random_state=RAND_SEED
)
X_test_ul, _ = make_example_unequal_length(
    n_cases=5, max_n_timepoints=15, min_n_timepoints=10, random_state=RAND_SEED
)


class ClassifierFitPredict(ClassifierTestScenario):
    """Fit/predict with univariate X, labels y."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": False,
        "is_enabled": True,
        "n_classes": 2,
    }

    args = {
        "fit": {"y": y, "X": X},
        "predict": {"X": X_test},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


class ClassifierFitPredictMultivariate(ClassifierTestScenario):
    """Fit/predict with multivariate panel X and labels y."""

    _tags = {
        "X_univariate": False,
        "X_unequal_length": False,
        "is_enabled": True,
        "n_classes": 2,
    }

    args = {
        "fit": {"y": y, "X": X_mv},
        "predict": {"X": X_test_mv},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


class ClassifierFitPredictUnequalLength(ClassifierTestScenario):
    """Fit/predict with univariate X and labels y, unequal length series."""

    _tags = {
        "X_univariate": True,
        "X_unequal_length": True,
        "is_enabled": True,
        "n_classes": 2,
    }

    args = {
        "fit": {"y": y, "X": X_ul},
        "predict": {"X": X_ul},
    }
    default_method_sequence = ["fit", "predict", "predict_proba", "decision_function"]
    default_arg_sequence = ["fit", "predict", "predict", "predict"]


scenarios_classification = [
    ClassifierFitPredict,
    ClassifierFitPredictMultivariate,
    ClassifierFitPredictUnequalLength,
]

# same scenarios used for early classification
scenarios_early_classification = [
    ClassifierFitPredict,
    ClassifierFitPredictMultivariate,
    ClassifierFitPredictUnequalLength,
]

# we use the same scenarios for regression, as in the old test suite
scenarios_regression = [
    ClassifierFitPredict,
    ClassifierFitPredictMultivariate,
    ClassifierFitPredictUnequalLength,
]
