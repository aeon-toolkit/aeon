"""Tests for BaseSimilaritySearch."""

__maintainer__ = ["baraline"]

from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

from aeon.similarity_search import BaseSimiliaritySearch
from aeon.testing.utils.data_gen import (
    make_example_3d_numpy,
    make_example_unequal_length,
)


class BaseInstance(BaseSimiliaritySearch):
    """Mock class for BaseSimiliaritySearch."""

    def __init__(
        self,
        distance="euclidean",
        distance_args=None,
        normalize=False,
        speed_up="fastest",
        n_jobs=1,
        query_length=5,
    ):
        self.query_length = query_length
        super().__init__(
            distance=distance,
            distance_args=distance_args,
            normalize=normalize,
            speed_up=speed_up,
            n_jobs=n_jobs,
        )

    def _fit(self, X, y=None):
        """Mock function for _fit."""
        self._store_mean_std_from_inputs(5)
        return self

    def get_speedup_function_names(self):
        """Mock function for get_speedup_function_names."""
        return {}


def test_fit():
    """Test fit function with mean and std computation."""
    X = make_example_3d_numpy(return_y=False, n_timepoints=8, n_channels=2)
    query_length = 5
    estimator = BaseInstance(query_length=query_length).fit(X)
    assert_array_equal(X, estimator.X_)
    for i in range(len(X)):
        for j in range(X[i].shape[1] - query_length + 1):
            subsequence = X[i, :, j : j + query_length]
            assert_almost_equal(estimator.X_means_[i][:, j], subsequence.mean(axis=-1))
            assert_almost_equal(estimator.X_stds_[i][:, j], subsequence.std(axis=-1))


def test_fit_unequal():
    """Test fit function with mean and std computation for unequal length."""
    X = make_example_unequal_length(
        return_y=False, min_n_timepoints=8, max_n_timepoints=10, n_channels=2
    )
    query_length = 5
    estimator = BaseInstance(query_length=query_length).fit(X)
    assert_equal(len(X), len(estimator.X_))
    for i in range(len(X)):
        assert_array_equal(X[i], estimator.X_[i])
        for j in range(X[i].shape[1] - query_length + 1):
            subsequence = X[i][:, j : j + query_length]
            assert_almost_equal(estimator.X_means_[i][:, j], subsequence.mean(axis=-1))
            assert_almost_equal(estimator.X_stds_[i][:, j], subsequence.std(axis=-1))
