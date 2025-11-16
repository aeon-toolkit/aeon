"""Tests for MechaClassifier."""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from aeon.classification.feature_based._mecha import (
    MechaClassifier,
    _adaptive_saving_features,
    _gwo,
    _objective_function,
)
from aeon.datasets import load_basic_motions

GWO_DIM = 1
GWO_SEARCH_SPACE = [1.0, 3.0]
GWO_DOWN_RATE = 4


@pytest.fixture
def mecha_test_data():
    """Load minimal data for Mecha testing and convert string labels to int."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    N_CASES_PER_CLASS = 2
    first_train = np.where(y_train == 0)[0][:N_CASES_PER_CLASS]
    second_train = np.where(y_train == 1)[0][:N_CASES_PER_CLASS]
    train_indices = np.concatenate([first_train, second_train])
    first_test = np.where(y_test == 0)[0][:N_CASES_PER_CLASS]
    second_test = np.where(y_test == 1)[0][:N_CASES_PER_CLASS]
    test_indices = np.concatenate([first_test, second_test])
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    return X_train, y_train, X_test, y_test


def test_gwo_output_format():
    """Test the Grey Wolf Optimizer returns correct format."""
    X = np.random.random(size=(5, 1, 10))
    y = np.array([0, 1, 0, 1, 0])
    k1_pos, k1_score = _gwo(
        _objective_function,
        X,
        y,
        dim=GWO_DIM,
        search_space=GWO_SEARCH_SPACE,
        down_rate=GWO_DOWN_RATE,
        max_iter=1,
        num_wolves=2,
        seed=0,
    )
    assert isinstance(k1_pos, np.ndarray)
    assert k1_pos.shape == (GWO_DIM,)
    assert isinstance(k1_score, float)
    assert GWO_SEARCH_SPACE[0] <= k1_pos[0] <= GWO_SEARCH_SPACE[1]


def test_adaptive_saving_features_output_format():
    """Test adaptive feature selection returns a list of integer counts."""
    n_features = 20
    n_thresholds = 3
    trainX = np.random.random(size=(10, n_features))
    scoresList = [np.random.random(n_features), np.random.random(n_features)]
    thresholds = [1e-4, 5e-4, 1e-3]
    kList = _adaptive_saving_features(trainX, scoresList, thresholds)
    assert isinstance(kList, np.ndarray)
    assert kList.shape == (n_thresholds,)
    assert all(kList >= 1)
    assert all(kList.astype(int) == kList)


def test_mecha_classifier_fit(mecha_test_data):
    """Test MechaClassifier can fit without error."""
    X_train, y_train, _, _ = mecha_test_data
    clf = MechaClassifier(
        max_iter=1, num_wolves=2, max_rate=4, basic_extractor="Catch22"
    )
    clf.fit(X_train, y_train)
    assert clf.optimized_k1 is not None
    assert clf.scaler is not None
    assert len(clf.clfListExtraMI) > 0


def test_mecha_classifier_predict(mecha_test_data):
    """Test MechaClassifier can predict and returns correct shape."""
    X_train, y_train, X_test, y_test = mecha_test_data
    clf = MechaClassifier(max_iter=1, num_wolves=2, max_rate=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert y_pred.dtype == y_test.dtype


def test_mecha_classifier_predict_proba(mecha_test_data):
    """Test MechaClassifier predict_proba returns correct shape and properties."""
    X_train, y_train, X_test, y_test = mecha_test_data
    clf = MechaClassifier(max_iter=1, num_wolves=2, max_rate=4)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    n_classes = len(clf.classes_)
    n_cases = X_test.shape[0]
    assert probas.shape == (n_cases, n_classes)
    assert np.allclose(probas.sum(axis=1), 1.0)
