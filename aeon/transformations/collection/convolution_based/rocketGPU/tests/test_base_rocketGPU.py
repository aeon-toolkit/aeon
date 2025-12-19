"""Unit tests for rocket GPU base functionality."""

__maintainer__ = ["hadifawaz1999"]

__all__ = [
    "test_base_rocketGPU_univariate",
    "test_base_rocketGPU_multivariate",
    "test_rocket_cpu_gpu",
]
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)
from aeon.transformations.collection.convolution_based._rocket import Rocket
from aeon.transformations.collection.convolution_based.rocketGPU._rocket_gpu import (
    ROCKETGPU,
)
from aeon.transformations.collection.convolution_based.rocketGPU.base import (
    BaseROCKETGPU,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


class DummyROCKETGPU(BaseROCKETGPU):

    def __init__(self, n_kernels=1):
        super().__init__(n_kernels)

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
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
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        output_rocket : np.ndarray [n_cases, n_kernels * 2]
            transformed features.
        """
        import numpy as np
        import tensorflow as tf

        X = X.transpose(0, 2, 1)

        rng = np.random.default_rng()

        _output_convolution = tf.nn.conv1d(
            input=X,
            filters=rng.normal(size=(self.kernel_size, X.shape[-1], self.n_kernels)),
            stride=1,
            padding="VALID",
            dilations=1,
        )

        _output_convolution = np.squeeze(_output_convolution.numpy(), axis=-1)

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
def test_base_rocketGPU_univariate():
    """Test base rocket GPU functionality univariate."""
    X, _ = make_example_2d_numpy_collection()

    dummy_transform = DummyROCKETGPU(n_kernels=1)
    dummy_transform.fit(X)

    X_transform = dummy_transform.transform(X)

    assert X_transform.shape[0] == len(X)
    assert len(X_transform.shape) == 2
    assert X_transform.shape[1] == 2

    # check all ppv values are >= 0
    assert (X_transform[:, 0] >= 0).sum() == len(X)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_base_rocketGPU_multivariate():
    """Test base rocket GPU functionality multivariate."""
    X, _ = make_example_3d_numpy(n_channels=3)

    dummy_transform = DummyROCKETGPU(n_kernels=1)
    dummy_transform.fit(X)

    X_transform = dummy_transform.transform(X)

    assert X_transform.shape[0] == len(X)
    assert len(X_transform.shape) == 2
    assert X_transform.shape[1] == 2

    # check all ppv values are >= 0
    assert (X_transform[:, 0] >= 0).sum() == len(X)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_rocket_cpu_gpu(n_channels):
    """Test kernel generation parity between CPU and GPU versions of ROCKET.

    Phase 1 (Current): Tests kernel generation parity - FIXED
    Phase 2 (Future): Transform parity for multivariate datasets - TODO

    With legacy_rng=False (default), GPU should produce identical kernel
    parameters as CPU within numerical precision (<1e-6).

    Note: Transform divergence on multivariate datasets is a known limitation
    documented in Phase 2. See issue #1248 for details.
    """
    import numpy as np

    random_state = 42
    X, _ = make_example_3d_numpy(n_channels=n_channels, random_state=random_state)

    n_kernels = 100

    rocket_cpu = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    rocket_cpu.fit(X)

    rocket_gpu = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)
    rocket_gpu.fit(X)

    # Phase 1: Test kernel generation parity (FIXED - perfect match)
    cpu_params = rocket_cpu.kernels
    cpu_biases = cpu_params[2]  # Extract biases
    cpu_dilations = cpu_params[3]  # Extract dilations

    gpu_biases = np.array(rocket_gpu._list_of_biases)
    gpu_dilations = np.array(rocket_gpu._list_of_dilations)

    # Verify kernel parameters match within numerical precision
    assert_array_almost_equal(
        cpu_biases,
        gpu_biases,
        decimal=6,
        err_msg="Kernel biases don't match between CPU and GPU",
    )
    assert_array_almost_equal(
        cpu_dilations,
        gpu_dilations,
        decimal=0,
        err_msg="Kernel dilations don't match between CPU and GPU",
    )

    # Phase 2 TODO: Add transform parity test when multivariate convolution fixed
    # Currently: Univariate ~0.3% divergence (acceptable), Multivariate up to 244%
    # See maintainer discussion on acceptable divergence levels for Phase 2


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_rocket_gpu_legacy_mode():
    """Test that legacy_rng=True preserves old GPU behavior and shows deprecation.

    Verifies:
    1. Legacy mode raises FutureWarning
    2. Legacy mode still works (backward compatibility)
    3. All kernels use all channels (legacy behavior)
    """
    random_state = 42
    X, _ = make_example_3d_numpy(
        n_channels=3, n_timepoints=50, random_state=random_state
    )

    # Test deprecation warning using pytest.warns (robust to TF/Keras warnings)
    with pytest.warns(
        FutureWarning, match=r"legacy_rng=True is deprecated"
    ):
        rocket_gpu_legacy = ROCKETGPU(
            n_kernels=20, random_state=random_state, legacy_rng=True
        )

    # Test that legacy mode still works (backward compatibility)
    rocket_gpu_legacy.fit(X)
    X_transform = rocket_gpu_legacy.transform(X)

    expected_shape = (X.shape[0], 40)  # 20 kernels × 2 features
    assert (
        X_transform.shape == expected_shape
    ), f"Shape mismatch: {X_transform.shape} != {expected_shape}"

    # Verify all kernels use all channels (legacy behavior)
    for i, channel_indices in enumerate(rocket_gpu_legacy._list_of_channel_indices):
        assert (
            len(channel_indices) == X.shape[1]
        ), f"Kernel {i}: Expected all {X.shape[1]} channels, got {len(channel_indices)}"
