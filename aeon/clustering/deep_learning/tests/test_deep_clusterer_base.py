"""Unit tests for clusterer deep learning base class functionality."""

import tempfile

import numpy as np
import pytest

from aeon.testing.mock_estimators import MockDeepClusterer
from aeon.testing.utils.data_gen import make_example_2d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = []


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_base_deep_clusterer():
    """Test base deep clusterer."""
    with tempfile.TemporaryDirectory() as tmp:
        last_file_name = "temp"
        # create a dummy deep classifier
        dummy_deep_clr = MockDeepClusterer(last_file_name=last_file_name)
        # generate random data
        X, y = make_example_2d_numpy()
        # test fit function on random data
        dummy_deep_clr.fit(X=X)
        # test save last model to file than delete it
        dummy_deep_clr.save_last_model_to_file(file_path=tmp)

        # test summary of model
        assert dummy_deep_clr.summary() is not None
        ypred = dummy_deep_clr.predict(X)
        assert ypred is not None
        assert len(ypred) == len(y)
        ypred_proba = dummy_deep_clr.predict_proba(X)
        assert ypred_proba is not None
        assert len(ypred_proba[0]) == len(np.unique(y))
        score = dummy_deep_clr.score(X)
        assert isinstance(score, np.float64)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("algorithm", ["kmeans", "kshape", "kmedoids"])
def test_base_deep_clusterer_with_algorithm(algorithm):
    """Test base deep clusterer with different algorithms."""
    with tempfile.TemporaryDirectory() as tmp:
        last_file_name = "temp"
        # create a dummy deep classifier
        dummy_deep_clr = MockDeepClusterer(last_file_name=last_file_name)
        # set the clustering algorithm
        dummy_deep_clr.clustering_algorithm = algorithm
        # generate random data
        X, y = make_example_2d_numpy()
        # test fit function on random data
        dummy_deep_clr.fit(X=X)
        # test save last model to file than delete it
        dummy_deep_clr.save_last_model_to_file(file_path=tmp)

        # test summary of model
        assert dummy_deep_clr.summary() is not None
        ypred = dummy_deep_clr.predict(X)
        assert ypred is not None
        assert len(ypred) == len(y)
        ypred_proba = dummy_deep_clr.predict_proba(X)
        assert ypred_proba is not None
        assert len(ypred_proba[0]) == len(np.unique(y))
        score = dummy_deep_clr.score(X)
        assert isinstance(score, np.float64)
