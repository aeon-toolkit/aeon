# -*- coding: utf-8 -*-
import pytest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from aeon.datasets import make_example_3d_numpy
from aeon.distances._distance import DISTANCES


@pytest.mark.parametrize("dist", DISTANCES)
def test_function_transformer(dist):
    """Test all distances work with FunctionTransformer in a pipeline."""
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=10)
    distance = dist["distance"]
    pairwise = dist["pairwise_distance"]
    ft = FunctionTransformer(pairwise)
    X2 = ft.transform(X)
    assert X2.shape[0] == X.shape[0]  # same number of cases
    assert X2.shape[1] == X.shape[0]  # pairwise
    assert X2[0][0] == 0  # identify should always be zero
    d1 = distance(X[0], X[1])
    assert X2[0][1] == d1
    d2 = distance(X[-1], X[-2])
    assert X2[-1][-2] == d2


@pytest.mark.parametrize("dist", DISTANCES)
def test_knn(dist):
    """Test all distances work with KNN in a pipeline."""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, regression_target=True, return_y=True
    )
    pairwise = dist["pairwise_distance"]
    ft = FunctionTransformer(pairwise)
    pipe = Pipeline(steps=[("FunctionTransformer", ft), ("KNN", KNeighborsRegressor())])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


@pytest.mark.parametrize("dist", DISTANCES)
def test_support_vector_machine(dist):
    """Test all distances work with DBSCAN"""
    X, y = make_example_3d_numpy(
        n_cases=5, n_channels=1, n_timepoints=10, return_y=True
    )
    pairwise = dist["pairwise_distance"]
    ft = FunctionTransformer(pairwise)
    pipe = Pipeline(steps=[("FunctionTransformer", ft), ("SVM", SVC())])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)


@pytest.mark.parametrize("dist", DISTANCES)
def test_clusterer(dist):
    """Test all distances work with SVM in a pipeline."""
    X = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=10)
    db = DBSCAN(metric="precomputed", eps=2.5)
    preds = db.fit_predict(dist["pairwise_distance"](X))
    assert len(preds) == len(X)
