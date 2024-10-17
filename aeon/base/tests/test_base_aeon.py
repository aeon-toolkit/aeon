"""Tests for universal base class that require aeon or sklearn imports."""

__maintainer__ = []

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from aeon.classification.feature_based import SummaryClassifier
from aeon.pipeline import make_pipeline
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import Tabularizer


def test_get_fitted_params_sklearn():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj aeon component returns expected nested params
    """
    X, y = make_example_3d_numpy()
    clf = SummaryClassifier(estimator=DecisionTreeClassifier())
    clf.fit(X, y)

    # params = clf.get_fitted_params()

    # todo v1.0.0 fix this


def test_get_fitted_params_sklearn_nested():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj aeon component returns expected nested params
    """
    X, y = make_example_3d_numpy()
    pipe = make_pipeline(Tabularizer(), StandardScaler(), DecisionTreeClassifier())
    clf = SummaryClassifier(estimator=pipe)
    clf.fit(X, y)

    # params = clf.get_fitted_params()

    # todo v1.0.0 fix this
