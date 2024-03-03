"""Unit tests for rocket GPU base functionality."""

__maintainer__ = ["hadifawaz1999", "AnonymousCodes911"]

__all__ = ["test_base_rocket_univariate", "test_base_rocket_multivariate"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.testing.utils.data_gen import make_example_2d_numpy, make_example_3d_numpy
from aeon.transformations.collection.convolution_based._rocket import Rocket
from aeon.transformations.collection.convolution_based.rocketGPU.base import (
    BaseROCKETGPU,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.fixture
def random_seed():
    return 10


class DummyROCKET(Rocket):

    def __init__(self, n_filters=1, random_seed=None):
        super().__init__(n_filters)
        self.random_seed = random_seed
        self.n_filters = n_filters

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_instances, n_channels, n_timepoints)
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        self
        """

        self.kernel_size = 2

    def _get_ppv(self, x):
        import tensorflow as tf

        x_pos = tf.math.count_nonzero(tf.nn.relu(x), axis=1)
        return tf.math.divide(x_pos, x.shape[1])

    def _get_max(self, x):
        import tensorflow as tf

        return tf.math.reduce_max(x, axis=1)

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, n_timepoints]
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        output_rocket : np.ndarray [n_instances, n_filters * 2]
            transformed features.
        """
        X = X.transpose(0, 2, 1)

        _output_convolution = np.random.normal(
            size=(self.kernel_size, X.shape[-1], self.n_filters)
        )

        _output_convolution = np.squeeze(_output_convolution, axis=-1)

        _ppv = self._get_ppv(x=_output_convolution)
        _max = self._get_max(x=_output_convolution)

        _output_features = np.concatenate(
            (np.expand_dims(_ppv, axis=-1), np.expand_dims(_max, axis=-1)),
            axis=1,
        )

        return _output_features


class DummyROCKETGPU(BaseROCKETGPU):

    def __init__(self, n_filters=1, random_seed=None):
        super().__init__(n_filters)
        self.random_seed = random_seed

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_instances, n_channels, n_timepoints)
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        self
        """

        self.kernel_size = 2

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_instances, n_channels, n_timepoints]
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        output_rocket : np.ndarray [n_instances, n_filters * 2]
            transformed features.
        """
        X = X.transpose(0, 2, 1)

        _output_convolution = np.random.normal(
            size=(self.kernel_size, X.shape[-1], self.n_filters)
        )

        _output_convolution = np.squeeze(_output_convolution, axis=-1)

        _ppv = self._get_ppv(x=_output_convolution)
        _max = self._get_max(x=_output_convolution)

        _output_features = np.concatenate(
            (np.expand_dims(_ppv, axis=-1), np.expand_dims(_max, axis=-1)),
            axis=1,
        )

        return _output_features


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_base_rocket_univariate(random_seed):
    """Test base rocket functionality univariate."""
    X, _ = make_example_2d_numpy()

    dummy_transform_cpu = DummyROCKET(n_filters=1, random_seed=random_seed)
    dummy_transform_cpu.fit(X)
    X_transform_cpu = dummy_transform_cpu.transform(X)

    dummy_transform_gpu = DummyROCKETGPU(n_filters=1, random_seed=random_seed)
    dummy_transform_gpu.fit(X)
    X_transform_gpu = dummy_transform_gpu.transform(X)

    assert_array_almost_equal(X_transform_cpu, X_transform_gpu, decimal=8)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_base_rocket_multivariate(random_seed):
    """Test base rocket functionality multivariate."""
    X, _ = make_example_3d_numpy(n_channels=3)

    dummy_transform_cpu = DummyROCKET(n_filters=1, random_seed=random_seed)
    dummy_transform_cpu.fit(X)
    X_transform_cpu = dummy_transform_cpu.transform(X)

    dummy_transform_gpu = DummyROCKETGPU(n_filters=1, random_seed=random_seed)
    dummy_transform_gpu.fit(X)
    X_transform_gpu = dummy_transform_gpu.transform(X)

    assert_array_almost_equal(X_transform_cpu, X_transform_gpu, decimal=8)
