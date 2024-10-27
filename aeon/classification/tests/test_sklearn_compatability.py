"""Unit tests for aeon classifier compatability with sklearn interfaces."""

__maintainer__ = []
__all__ = [
    "test_sklearn_cross_validation",
    "test_sklearn_cross_validation_iterators",
    "test_sklearn_parameter_tuning",
    "test_sklearn_composite_classifiers",
]

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    RandomizedSearchCV,
    RepeatedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from aeon.classification.interval_based import CanonicalIntervalForestClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection import Resizer

# StratifiedGroupKFold(n_splits=2), removed because it is not available in sklearn 0.24
CROSS_VALIDATION_METHODS = [
    KFold(n_splits=2),
    RepeatedKFold(n_splits=2, n_repeats=2),
    LeaveOneOut(),
    LeavePOut(p=5),
    ShuffleSplit(n_splits=2, test_size=0.25),
    StratifiedKFold(n_splits=2),
    StratifiedShuffleSplit(n_splits=2, test_size=0.25),
    GroupKFold(n_splits=2),
    LeavePGroupsOut(n_groups=5),
    GroupShuffleSplit(n_splits=2, test_size=0.25),
    TimeSeriesSplit(n_splits=2),
]
PARAMETER_TUNING_METHODS = [
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
]
COMPOSITE_ESTIMATORS = [
    Pipeline(
        [
            ("transform", Resizer(length=10)),
            ("clf", CanonicalIntervalForestClassifier._create_test_instance()),
        ]
    ),
    VotingClassifier(
        estimators=[
            ("clf1", CanonicalIntervalForestClassifier._create_test_instance()),
            ("clf2", CanonicalIntervalForestClassifier._create_test_instance()),
            ("clf3", CanonicalIntervalForestClassifier._create_test_instance()),
        ]
    ),
    CalibratedClassifierCV(
        estimator=CanonicalIntervalForestClassifier._create_test_instance(),
        cv=2,
    ),
]


def test_sklearn_cross_validation():
    """Test sklearn cross-validation works with aeon data and classifiers."""
    clf = CanonicalIntervalForestClassifier._create_test_instance()
    X, y = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=30)
    scores = cross_val_score(clf, X, y=y, cv=KFold(n_splits=2))
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize("cross_validation_method", CROSS_VALIDATION_METHODS)
def test_sklearn_cross_validation_iterators(cross_validation_method):
    """Test if sklearn cross-validation iterators can handle aeon data."""
    X, y = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=30)
    groups = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]

    for train, test in cross_validation_method.split(X=X, y=y, groups=groups):
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)


@pytest.mark.parametrize("parameter_tuning_method", PARAMETER_TUNING_METHODS)
def test_sklearn_parameter_tuning(parameter_tuning_method):
    """Test if sklearn parameter tuners can handle aeon data and classifiers."""
    clf = CanonicalIntervalForestClassifier._create_test_instance()
    param_grid = {"n_intervals": [2, 3], "att_subsample_size": [2, 3]}
    X, y = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=30)

    parameter_tuning_method = parameter_tuning_method(
        clf, param_grid, cv=KFold(n_splits=2)
    )
    parameter_tuning_method.fit(X, y)
    assert isinstance(
        parameter_tuning_method.best_estimator_, CanonicalIntervalForestClassifier
    )


@pytest.mark.parametrize("composite_classifier", COMPOSITE_ESTIMATORS)
def test_sklearn_composite_classifiers(composite_classifier):
    """Test if sklearn composite classifiers can handle aeon data and classifiers."""
    X, y = make_example_3d_numpy(n_cases=20, n_channels=2, n_timepoints=30)
    composite_classifier.fit(X, y)
    preds = composite_classifier.predict(X=X)
    assert isinstance(preds, np.ndarray)
