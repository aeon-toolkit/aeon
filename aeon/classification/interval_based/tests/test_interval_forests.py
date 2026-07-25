"""Test interval forest classifiers."""

import warnings

import numpy as np
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
    params.update({"base_estimator": ContinuousIntervalTree(), "random_state": 0})

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


def test_tic_curves_all_stumps():
    """Test that an all-stump forest returns empty curves instead of raising.

    A constant input series gives the base estimators no useful split, so
    every tree in the forest is a stump with no internal nodes. In this case
    ``temporal_importance_curves`` should return empty results rather than
    raising a ValueError, and the plotting function should raise a clear
    error rather than silently plotting nothing.
    """
    X_train = np.zeros((10, 1, 20))
    y_train = np.array([0, 1] * 5)

    clf = CanonicalIntervalForestClassifier(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        base_estimator=ContinuousIntervalTree(),
        random_state=0,
    )
    clf.fit(X_train, y_train)

    names, curves = clf.temporal_importance_curves()
    assert names == []
    assert curves == []

    curves_dict = clf.temporal_importance_curves(return_dict=True)
    assert curves_dict == {}

    if _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"):
        import matplotlib

        matplotlib.use("Agg")

        with pytest.raises(ValueError, match="no splits"):
            plot_temporal_importance_curves(curves, names)


def test_drcif_warns_once_for_use_pycatch22():
    """Test that internal Catch22 clones do not repeat the public warning."""
    X_train, y_train = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]
    params = DrCIFClassifier._get_test_params()
    params["use_pycatch22"] = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        DrCIFClassifier(**params).fit(X_train, y_train)

    deprecation_warnings = [
        warning
        for warning in caught
        if "use_pycatch22" in str(warning.message)
        and issubclass(warning.category, FutureWarning)
    ]
    assert len(deprecation_warnings) == 1


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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        clf = cls(**params)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)

    deprecation_warnings = [
        warning for warning in caught if "use_pycatch22" in str(warning.message)
    ]
    assert len(deprecation_warnings) == 1
    _assert_predict_probabilities(prob, X_test, n_classes=2)
