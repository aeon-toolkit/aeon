"""Tests for REDCOMETS oversampling without imbalanced-learn."""

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from aeon.classification.dictionary_based import REDCOMETS
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.imbalance import RandomOverSampler, SMOTE
from aeon.utils.validation._dependencies import _check_soft_dependencies


def _normalise_2d(X2d):
    """Apply aeon collection Normalizer to a 2D panel and return 2D."""
    return Normalizer().fit_transform(X2d).squeeze(1)


def _redcomets_resample_aeon(X2d, y, random_state=0, n_jobs=1):
    """Mirror REDCOMETS oversampling with aeon transformers."""
    X = _normalise_2d(X2d)
    min_neighbours = min(Counter(y).values())
    max_neighbours = max(Counter(y).values())
    if min_neighbours == max_neighbours:
        return X, y
    if min_neighbours > 5:
        min_neighbours = 6
    n_neighbors = min_neighbours - 2
    X_3d = X[:, np.newaxis, :]
    try:
        if n_neighbors < 1:
            raise ValueError("not enough neighbours")
        X_smote, y_smote = SMOTE(
            n_neighbors=n_neighbors,
            random_state=random_state,
            distance="euclidean",
            n_jobs=n_jobs,
        ).fit_transform(X_3d, y)
    except (ValueError, IndexError):
        X_smote, y_smote = RandomOverSampler(random_state=random_state).fit_transform(
            X_3d, y
        )
    return np.squeeze(X_smote, 1), y_smote


def test_redcomets_smote_class_counts_match_imblearn():
    """Class counts and samples after aeon SMOTE match the previous imblearn path."""
    if not _check_soft_dependencies(
        "imbalanced-learn",
        package_import_alias={"imbalanced-learn": "imblearn"},
        severity="none",
    ):
        return

    from imblearn.over_sampling import SMOTE as ImbSMOTE

    rng = check_random_state(0)
    X2d = rng.randn(40, 30)
    y = np.array([0] * 24 + [1] * 16)
    Xn = _normalise_2d(X2d)
    min_neighbours = 6
    X_imb, y_imb = ImbSMOTE(
        sampling_strategy="all",
        k_neighbors=NearestNeighbors(n_neighbors=min_neighbours - 1),
        random_state=0,
    ).fit_resample(Xn, y)
    X_aeon, y_aeon = _redcomets_resample_aeon(X2d, y, random_state=0)

    assert Counter(map(str, y_imb)) == Counter(map(str, y_aeon))
    assert X_imb.shape == X_aeon.shape

    def sort_xy(X, y):
        y = np.asarray(y).astype(str)
        idx = np.lexsort((X[:, 0], y))
        return X[idx], y[idx]

    Xi, yi = sort_xy(X_imb, y_imb)
    Xa, ya = sort_xy(X_aeon, y_aeon)
    assert np.array_equal(yi, ya)
    assert np.allclose(Xi, Xa)


def test_redcomets_fits_without_imblearn_tag():
    """REDCOMETS no longer declares an imblearn soft dependency."""
    clf = REDCOMETS(variant=1, n_trees=2, random_state=0)
    deps = clf.get_tag("python_dependencies", None)
    if deps is None:
        deps = []
    if isinstance(deps, str):
        deps = [deps]
    assert "imblearn" not in deps
    assert "imbalanced-learn" not in deps


def test_redcomets_ros_fallback_on_tiny_minority():
    """When SMOTE cannot run, RandomOverSampler balances class counts."""
    rng = check_random_state(0)
    # one class with a single sample forces SMOTE neighbour failure
    X2 = rng.randn(10, 20)
    y = np.array([0] * 8 + [1] * 1 + [2] * 1)
    X_res, y_res = _redcomets_resample_aeon(X2, y, random_state=0)
    counts = Counter(map(int, y_res))
    assert len(set(counts.values())) == 1
    assert counts[0] == 8


def test_redcomets_end_to_end_fit_predict():
    """REDCOMETS fits and predicts on a small imbalanced panel."""
    rng = check_random_state(0)
    X = rng.randn(30, 1, 24)
    y = np.array([0] * 20 + [1] * 10)
    clf = REDCOMETS(variant=1, n_trees=5, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (30,)
