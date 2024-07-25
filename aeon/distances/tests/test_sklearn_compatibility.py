"""Tests for compatibility with sklearn."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_basic_motions, load_unit_test
from aeon.distances._distance import DISTANCES
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize("dist", DISTANCES)
def test_function_transformer(dist):
    """Test all distances work with FunctionTransformer in a pipeline."""
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=10, return_y=False)
    distance = dist["distance"]
    pairwise = dist["pairwise_distance"]
    ft = FunctionTransformer(pairwise)
    X2 = ft.transform(X)
    assert X2.shape[0] == X.shape[0]  # same number of cases
    assert X2.shape[1] == X.shape[0]  # pairwise
    assert X2[0][0] == 0  # identify should always be zero
    d1 = distance(X[0], X[1])
    assert_almost_equal(X2[0][1], d1)
    d2 = distance(X[-1], X[-2])
    assert_almost_equal(X2[-1][-2], d2)


@pytest.mark.parametrize("dist", DISTANCES)
def test_distance_based(dist):
    """Test all distances work with KNN in a pipeline."""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, regression_target=True
    )
    X2, y2 = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=10, regression_target=True
    )
    pairwise = dist["pairwise_distance"]
    Xt = pairwise(X)
    Xt2 = pairwise(X2, X)
    cls1 = KNeighborsRegressor(metric="precomputed")
    cls2 = SVR(kernel="precomputed")
    cls1.fit(Xt, y)
    preds = cls1.predict(Xt2)
    assert len(preds) == len(y2)
    cls2.fit(Xt, y)
    preds = cls2.predict(Xt2)
    assert len(preds) == len(y2)


@pytest.mark.parametrize("dist", DISTANCES)
def test_clusterer(dist):
    """Test all distances work with DBSCAN."""
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=10, return_y=False)
    db = DBSCAN(metric="precomputed", eps=2.5)
    preds = db.fit_predict(dist["pairwise_distance"](X))
    assert len(preds) == len(X)


@pytest.mark.parametrize("dist", DISTANCES)
@pytest.mark.parametrize("k", [1, 5])
def test_classification_univariate(dist, k):
    """Test all distances work with sklearn nearest neighbours."""
    # Test univariate with 2D format (compatible with sklearn)

    # Load the unit test dataset as a 2D numpy array
    X_train, y_train = load_unit_test(split="train", return_type="numpy2D")
    X_test, y_test = load_unit_test(split="test", return_type="numpy2D")
    indices = np.random.RandomState(0).choice(
        min(len(y_test), len(y_train)), 10, replace=False
    )
    # Compute the pairwise distance matrix for working with sklearn knn with
    # precomputed distances.
    X_train_precomputed_distance = dist["pairwise_distance"](X_train[indices])
    X_test_precomputed_distance = dist["pairwise_distance"](
        X_test[indices], X_train[indices]
    )

    knn_aeon = KNeighborsTimeSeriesClassifier(
        distance=dist["name"], n_neighbors=k
    )  # aeon
    knn_sk = KNeighborsClassifier(metric=dist["distance"], n_neighbors=k)  # sklearn
    knn_sk_precomputed = KNeighborsClassifier(
        metric="precomputed", n_neighbors=k
    )  # sklearn pre

    knn_aeon.fit(X_train[indices], y_train[indices])
    knn_sk.fit(X_train[indices], y_train[indices])
    knn_sk_precomputed.fit(X_train_precomputed_distance, y_train[indices])

    knn_aeon_probas = knn_aeon.predict_proba(X_test[indices])
    knn_sk_probas = knn_sk.predict_proba(X_test[indices])
    knn_sk_precomputed_probas = knn_sk_precomputed.predict_proba(
        X_test_precomputed_distance
    )

    assert_almost_equal(knn_aeon_probas, knn_sk_probas)
    assert_almost_equal(knn_aeon_probas, knn_sk_precomputed_probas)


@pytest.mark.parametrize("dist", DISTANCES)
@pytest.mark.parametrize("k", [1, 5])
def test_classification_multivariate(dist, k):
    """Test all distances work with sklearn nearest neighbours."""
    # Test multivariate dataset in two ways: A) concatenating channels to be compatible
    # with sklearn, and B) precomputing distances.

    # Load the basic motions dataset as a 3D numpy array
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3D")
    X_test, y_test = load_basic_motions(split="test", return_type="numpy3D")

    # Transform to 2D format concatenating channels.
    X_train_concat = X_train.reshape(X_train.shape[0], -1)
    X_test_concat = X_test.reshape(X_test.shape[0], -1)

    indices = np.random.RandomState(4).choice(
        min(len(y_test), len(y_train)), 10, replace=False
    )

    # A) Test multivariate with 2D format (concatenates channels to be compatible with
    # sklearn)
    knn_aeon = KNeighborsTimeSeriesClassifier(
        distance=dist["name"], n_neighbors=k
    )  # aeon concat
    knn_sk = KNeighborsClassifier(
        metric=dist["distance"], n_neighbors=k
    )  # sklearn concat

    knn_aeon.fit(X_train_concat[indices], y_train[indices])
    knn_sk.fit(X_train_concat[indices], y_train[indices])

    knn_aeon_probas = knn_aeon.predict_proba(X_test_concat[indices])
    knn_sk_probas = knn_sk.predict_proba(X_test_concat[indices])

    assert_almost_equal(knn_aeon_probas, knn_sk_probas)

    # ---------------------------------------------------------------------------------
    # B) Test multivariate with 3D format (compatible with sklearn if using precomputed
    # distances)

    # Compute the pairwise distance matrix
    X_train_precomputed_distance = dist["pairwise_distance"](X_train[indices])
    X_test_precomputed_distance = dist["pairwise_distance"](
        X_test[indices], X_train[indices]
    )

    knn_aeon_3D = KNeighborsTimeSeriesClassifier(
        distance=dist["name"], n_neighbors=k
    )  # aeon
    knn_sk_precomputed = KNeighborsClassifier(
        metric="precomputed", n_neighbors=k
    )  # sklearn precomputed

    knn_aeon_3D.fit(X_train[indices], y_train[indices])
    knn_sk_precomputed.fit(X_train_precomputed_distance, y_train[indices])

    knn_aeon_3D_probas = knn_aeon_3D.predict_proba(X_test[indices])
    knn_sk_precomputed_probas = knn_sk_precomputed.predict_proba(
        X_test_precomputed_distance
    )

    assert_almost_equal(knn_aeon_3D_probas, knn_sk_precomputed_probas)
