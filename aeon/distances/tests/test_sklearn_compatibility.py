"""Tests for compatibility with sklearn."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.distances._distance import DISTANCES, MIN_DISTANCES, MP_DISTANCES
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.testing.data_generation import make_example_3d_numpy


@pytest.mark.parametrize("dist", DISTANCES)
def test_function_transformer(dist):
    """Test all distances work with FunctionTransformer in a pipeline."""
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    X = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, return_y=False, random_state=1
    )
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
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    X, y = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=10, regression_target=True
    )
    X2, y2 = make_example_3d_numpy(
        n_cases=6, n_channels=1, n_timepoints=10, regression_target=True
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
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=10, return_y=False)
    db = DBSCAN(metric="precomputed", eps=2.5)
    preds = db.fit_predict(dist["pairwise_distance"](X))
    assert len(preds) == len(X)


@pytest.mark.parametrize("dist", DISTANCES)
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(
    "task",
    [
        ["cls", KNeighborsClassifier, KNeighborsTimeSeriesClassifier],
        ["reg", KNeighborsRegressor, KNeighborsTimeSeriesRegressor],
    ],
)
def test_univariate(dist, k, task):
    """Test all distances work with sklearn nearest neighbours."""
    # TODO: when solved the issue with lcss and edr, remove this condition
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    # https://github.com/aeon-toolkit/aeon/issues/882

    if dist["name"] in ["lcss", "edr"]:
        return

    # Test univariate with 2D format (compatible with sklearn)
    problem_type, knn_sk_func, knn_aeon_func = task

    # Create a collection as a 2D numpy array
    if problem_type == "cls":
        reg = False
    else:
        reg = True
    X_train, y_train = make_example_3d_numpy(
        random_state=0, n_cases=6, regression_target=reg
    )
    X_test, y_test = make_example_3d_numpy(
        random_state=2, n_cases=6, regression_target=reg
    )
    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    # Compute the pairwise distance matrix for working with sklearn knn with
    # precomputed distances.
    X_train_precomp_distance = dist["pairwise_distance"](X_train)
    X_test_precomp_distance = dist["pairwise_distance"](X_test, X_train)

    knn_aeon = knn_aeon_func(distance=dist["name"], n_neighbors=k)  # aeon
    knn_sk = knn_sk_func(metric=dist["distance"], n_neighbors=k)  # sklearn
    knn_sk_precomp = knn_sk_func(metric="precomputed", n_neighbors=k)  # sklearn pre

    knn_aeon.fit(X_train, y_train)
    knn_sk.fit(X_train, y_train)
    knn_sk_precomp.fit(X_train_precomp_distance, y_train)

    if problem_type == "cls":
        knn_aeon_output = knn_aeon.predict_proba(X_test)
        knn_sk_output = knn_sk.predict_proba(X_test)
        knn_sk_precomp_output = knn_sk_precomp.predict_proba(X_test_precomp_distance)
    elif problem_type == "reg":
        knn_aeon_output = knn_aeon.predict(X_test)
        knn_sk_output = knn_sk.predict(X_test)
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
            KNeighborsClassifier,
            KNeighborsTimeSeriesClassifier,
        ],
        [
            "reg",
            KNeighborsRegressor,
            KNeighborsTimeSeriesRegressor,
        ],
    ],
)
def test_multivariate(dist, k, task):
    """Test all distances work with sklearn nearest neighbours."""
    # Skip for now
    if dist["name"] in MIN_DISTANCES or dist["name"] in MP_DISTANCES:
        return
    # TODO: when solved the issue with lcss and edr, remove this condition
    # https://github.com/aeon-toolkit/aeon/issues/882
    if dist["name"] in ["lcss", "edr"]:
        return

    # Test multivariate dataset in two ways: A) concatenating channels to be compatible
    # with sklearn, and B) precomputing distances.

    problem_type, knn_sk_func, knn_aeon_func = task
    if problem_type == "cls":
        reg = False
    else:
        reg = True
    X_train, y_train = make_example_3d_numpy(
        random_state=0, n_cases=6, regression_target=reg
    )
    X_test, y_test = make_example_3d_numpy(
        random_state=2, n_cases=6, regression_target=reg
    )

    # Transform to 2D format concatenating channels.
    X_train_concat = X_train.reshape(X_train.shape[0], -1)
    X_test_concat = X_test.reshape(X_test.shape[0], -1)

    # A) Test multivariate with 2D format (concatenates channels to be compatible with
    # sklearn)
    knn_aeon = knn_aeon_func(distance=dist["name"], n_neighbors=k)  # aeon concat
    knn_sk = knn_sk_func(metric=dist["distance"], n_neighbors=k)  # sklearn concat

    knn_aeon.fit(X_train_concat, y_train)
    knn_sk.fit(X_train_concat, y_train)

    if problem_type == "cls":
        knn_aeon_output = knn_aeon.predict_proba(X_test_concat)
        knn_sk_output = knn_sk.predict_proba(X_test_concat)
    elif problem_type == "reg":
        knn_aeon_output = knn_aeon.predict(X_test_concat)
        knn_sk_output = knn_sk.predict(X_test_concat)

    assert_allclose(knn_aeon_output, knn_sk_output)

    # ---------------------------------------------------------------------------------
    # B) Test multivariate with 3D format (compatible with sklearn if using precomputed
    # distances)

    # Compute the pairwise distance matrix
    X_train_precomp_distance = dist["pairwise_distance"](X_train)
    X_test_precomp_distance = dist["pairwise_distance"](X_test, X_train)

    knn_aeon_3D = knn_aeon_func(distance=dist["name"], n_neighbors=k)  # aeon
    knn_sk_precomp = knn_sk_func(metric="precomputed", n_neighbors=k)  # sklearn precomp

    knn_aeon_3D.fit(X_train, y_train)
    knn_sk_precomp.fit(X_train_precomp_distance, y_train)

    if problem_type == "cls":
        knn_aeon_3D_output = knn_aeon_3D.predict_proba(X_test)
        knn_sk_precomp_output = knn_sk_precomp.predict_proba(X_test_precomp_distance)
    elif problem_type == "reg":
        knn_aeon_3D_output = knn_aeon_3D.predict(X_test)
        knn_sk_precomp_output = knn_sk_precomp.predict(X_test_precomp_distance)

    assert_allclose(knn_aeon_3D_output, knn_sk_precomp_output)
