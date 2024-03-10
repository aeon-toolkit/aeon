"""Tests for compatibility with sklearn."""

import pytest
from numpy.testing import assert_almost_equal
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR

from aeon.distances._distance import DISTANCES
from aeon.testing.utils.data_gen import make_example_3d_numpy


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
