"""Tests for compatibility with sklearn."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import (
    load_basic_motions,
    load_cardano_sentiment,
    load_covid_3month,
    load_unit_test,
)
from aeon.distances._distance import DISTANCES
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
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
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(
    "task",
    [
        ["cls", load_unit_test, KNeighborsClassifier, KNeighborsTimeSeriesClassifier],
        ["reg", load_covid_3month, KNeighborsRegressor, KNeighborsTimeSeriesRegressor],
    ],
)
def test_univariate(dist, k, task):
    """Test all distances work with sklearn nearest neighbours."""
    # TODO: when solved the issue with lcss and edr, remove this condition
    # https://github.com/aeon-toolkit/aeon/issues/882
    if dist["name"] in ["lcss", "edr"]:
        return

    # Test univariate with 2D format (compatible with sklearn)
    problem_type, problem_loader, knn_sk_func, knn_aeon_func = task

    # Load the unit test dataset as a 2D numpy array
    X_train, y_train = problem_loader(split="train", return_type="numpy2D")
    X_test, y_test = problem_loader(split="test", return_type="numpy2D")
    indices = np.random.RandomState(0).choice(
        min(len(y_test), len(y_train)), 6, replace=False
    )
    # Compute the pairwise distance matrix for working with sklearn knn with
    # precomputed distances.
    X_train_precomp_distance = dist["pairwise_distance"](X_train[indices])
    X_test_precomp_distance = dist["pairwise_distance"](
        X_test[indices], X_train[indices]
    )

    knn_aeon = knn_aeon_func(distance=dist["name"], n_neighbors=k)  # aeon
    knn_sk = knn_sk_func(metric=dist["distance"], n_neighbors=k)  # sklearn
    knn_sk_precomp = knn_sk_func(metric="precomputed", n_neighbors=k)  # sklearn pre

    knn_aeon.fit(X_train[indices], y_train[indices])
    knn_sk.fit(X_train[indices], y_train[indices])
    knn_sk_precomp.fit(X_train_precomp_distance, y_train[indices])

    if problem_type == "cls":
        knn_aeon_output = knn_aeon.predict_proba(X_test[indices])
        knn_sk_output = knn_sk.predict_proba(X_test[indices])
        knn_sk_precomp_output = knn_sk_precomp.predict_proba(X_test_precomp_distance)
    elif problem_type == "reg":
        knn_aeon_output = knn_aeon.predict(X_test[indices])
        knn_sk_output = knn_sk.predict(X_test[indices])
        knn_sk_precomp_output = knn_sk_precomp.predict(X_test_precomp_distance)

    assert_allclose(knn_aeon_output, knn_sk_output)
    assert_allclose(knn_aeon_output, knn_sk_precomp_output)


@pytest.mark.parametrize("dist", DISTANCES)
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(
    "task",
    [
        [
            "cls",
            load_basic_motions,
            KNeighborsClassifier,
            KNeighborsTimeSeriesClassifier,
        ],
        [
            "reg",
            load_cardano_sentiment,
            KNeighborsRegressor,
            KNeighborsTimeSeriesRegressor,
        ],
    ],
)
def test_multivariate(dist, k, task):
    """Test all distances work with sklearn nearest neighbours."""
    # TODO: when solved the issue with lcss and edr, remove this condition
    # https://github.com/aeon-toolkit/aeon/issues/882
    if dist["name"] in ["lcss", "edr"]:
        return

    # Test multivariate dataset in two ways: A) concatenating channels to be compatible
    # with sklearn, and B) precomputing distances.

    problem_type, problem_loader, knn_sk_func, knn_aeon_func = task

    # Load the basic motions dataset as a 3D numpy array
    X_train, y_train = problem_loader(split="train", return_type="numpy3D")
    X_test, y_test = problem_loader(split="test", return_type="numpy3D")

    # Transform to 2D format concatenating channels.
    X_train_concat = X_train.reshape(X_train.shape[0], -1)
    X_test_concat = X_test.reshape(X_test.shape[0], -1)

    indices = np.random.RandomState(0).choice(
        min(len(y_test), len(y_train)), 6, replace=False
    )

    # A) Test multivariate with 2D format (concatenates channels to be compatible with
    # sklearn)
    knn_aeon = knn_aeon_func(distance=dist["name"], n_neighbors=k)  # aeon concat
    knn_sk = knn_sk_func(metric=dist["distance"], n_neighbors=k)  # sklearn concat

    knn_aeon.fit(X_train_concat[indices], y_train[indices])
    knn_sk.fit(X_train_concat[indices], y_train[indices])

    if problem_type == "cls":
        knn_aeon_output = knn_aeon.predict_proba(X_test_concat[indices])
        knn_sk_output = knn_sk.predict_proba(X_test_concat[indices])
    elif problem_type == "reg":
        knn_aeon_output = knn_aeon.predict(X_test_concat[indices])
        knn_sk_output = knn_sk.predict(X_test_concat[indices])

    assert_allclose(knn_aeon_output, knn_sk_output)

    # ---------------------------------------------------------------------------------
    # B) Test multivariate with 3D format (compatible with sklearn if using precomputed
    # distances)

    # Compute the pairwise distance matrix
    X_train_precomp_distance = dist["pairwise_distance"](X_train[indices])
    X_test_precomp_distance = dist["pairwise_distance"](
        X_test[indices], X_train[indices]
    )

    knn_aeon_3D = knn_aeon_func(distance=dist["name"], n_neighbors=k)  # aeon
    knn_sk_precomp = knn_sk_func(metric="precomputed", n_neighbors=k)  # sklearn precomp

    knn_aeon_3D.fit(X_train[indices], y_train[indices])
    knn_sk_precomp.fit(X_train_precomp_distance, y_train[indices])

    if problem_type == "cls":
        knn_aeon_3D_output = knn_aeon_3D.predict_proba(X_test[indices])
        knn_sk_precomp_output = knn_sk_precomp.predict_proba(X_test_precomp_distance)
    elif problem_type == "reg":
        knn_aeon_3D_output = knn_aeon_3D.predict(X_test[indices])
        knn_sk_precomp_output = knn_sk_precomp.predict(X_test_precomp_distance)

    assert_allclose(knn_aeon_3D_output, knn_sk_precomp_output)
