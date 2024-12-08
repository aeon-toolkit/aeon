"""Test interval forest classifiers."""

import pytest

from aeon.classification.interval_based import (
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    RandomIntervalSpectralEnsembleClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION
from aeon.testing.utils.estimator_checks import _assert_predict_probabilities
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_temporal_importance_curves


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "cls",
    [
        CanonicalIntervalForestClassifier,
        DrCIFClassifier,
        SupervisedTimeSeriesForest,
        TimeSeriesForestClassifier,
    ],
)
def test_tic_curves(cls):
    """Test whether temporal_importance_curves runs without error."""
    import matplotlib

    matplotlib.use("Agg")

    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"base_estimator": ContinuousIntervalTree()})

    clf = cls(**params)
    clf.fit(X_train, y_train)

    names, curves = clf.temporal_importance_curves()
    plot_temporal_importance_curves(curves, names)


@pytest.mark.parametrize("cls", [RandomIntervalSpectralEnsembleClassifier])
def test_tic_curves_invalid(cls):
    """Test whether temporal_importance_curves raises an error."""
    clf = cls()
    with pytest.raises(
        NotImplementedError, match="No temporal importance curves available."
    ):
        clf.temporal_importance_curves()


@pytest.mark.skipif(
    not _check_soft_dependencies(["pycatch22"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("cls", [CanonicalIntervalForestClassifier, DrCIFClassifier])
def test_forest_pycatch22(cls):
    """Test whether the forest classifiers with pycatch22 run without error."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]
    X_test, _ = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["test"]

    params = cls._get_test_params()
    if isinstance(params, list):
        params = params[0]
    params.update({"use_pycatch22": True})

    clf = cls(**params)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)
    _assert_predict_probabilities(prob, X_test, n_classes=2)
