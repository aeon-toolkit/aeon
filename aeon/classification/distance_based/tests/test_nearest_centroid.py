"""Tests for NearestCentroidClassifier."""

import numpy as np
import pytest

from aeon.classification.distance_based import NearestCentroidClassifier
from aeon.testing.data_generation import make_example_3d_numpy


def _data(n_cases=10, n_channels=1, random_state=1):
    return make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=12,
        n_labels=2,
        random_state=random_state,
    )


def test_fit_sets_one_centroid_per_class():
    """``fit`` builds one centroid per class with the series shape."""
    X, y = _data(n_channels=2)
    clf = NearestCentroidClassifier(distance="euclidean", average_method="mean").fit(
        X, y
    )
    assert set(clf.classes_) == set(np.unique(y))
    assert clf.centroids_.shape == (len(np.unique(y)), 2, 12)
    assert np.all(np.isfinite(clf.centroids_))


def test_predict_returns_known_labels():
    """``predict`` returns labels drawn from the training classes."""
    X, y = _data()
    X_test, _ = _data(n_cases=5, random_state=2)
    clf = NearestCentroidClassifier(distance="euclidean", average_method="mean").fit(
        X, y
    )
    preds = clf.predict(X_test)
    assert preds.shape == (5,)
    assert set(preds).issubset(set(clf.classes_))


@pytest.mark.parametrize("distance", ["soft_dtw", "soft_msm"])
def test_soft_distance_auto_promotes_to_soft_averaging(distance):
    """A soft distance with the default ``average_method`` uses soft averaging."""
    X, y = _data()
    clf = NearestCentroidClassifier(distance=distance, distance_params={"gamma": 0.1})
    clf.fit(X, y)
    assert clf._average_method == "soft"
    assert np.all(np.isfinite(clf.centroids_))


def test_soft_distance_with_hard_averaging_raises():
    """Explicitly pairing a soft distance with non-soft averaging raises."""
    X, y = _data()
    with pytest.raises(ValueError, match="soft distance"):
        NearestCentroidClassifier(distance="soft_dtw", average_method="petitjean").fit(
            X, y
        )


def test_soft_averaging_with_hard_distance_raises():
    """``average_method='soft'`` with a non-soft distance raises."""
    X, y = _data()
    with pytest.raises(ValueError, match="requires a soft distance"):
        NearestCentroidClassifier(distance="dtw", average_method="soft").fit(X, y)


def test_default_average_method_is_dba_for_hard_distance():
    """The default ``average_method`` resolves to DBA (petitjean) for ``dtw``."""
    X, y = _data()
    clf = NearestCentroidClassifier(distance="dtw").fit(X, y)
    assert clf._average_method == "petitjean"


def test_mean_and_dba_give_different_centroids():
    """Mean and DBA averaging produce different centroids for warped data."""
    X, y = _data()
    mean_clf = NearestCentroidClassifier(distance="dtw", average_method="mean").fit(
        X, y
    )
    dba_clf = NearestCentroidClassifier(distance="dtw", average_method="petitjean").fit(
        X, y
    )
    assert not np.allclose(mean_clf.centroids_, dba_clf.centroids_)
