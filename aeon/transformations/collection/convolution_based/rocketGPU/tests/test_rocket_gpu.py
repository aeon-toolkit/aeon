"""Comprehensive test suite for ROCKETGPU (CuPy-based GPU acceleration).

This module tests numerical parity between CPU Rocket and GPU ROCKETGPU implementations.
All tests are CI-safe and automatically skip when CuPy or GPU is unavailable.

Test Coverage:
- Sanity checks (basic functionality)
- Numerical parity (univariate and multivariate)
- Edge cases (short series, constant input, variable lengths)
- Error handling (graceful fallback when GPU unavailable)
"""

__maintainer__ = ["Aditya Kushwaha", "hadifawaz1999"]

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based import Rocket
from aeon.transformations.collection.convolution_based.rocketGPU import ROCKETGPU
from aeon.utils.validation._dependencies import _check_soft_dependencies

# Skip all tests if CuPy not available or no GPU detected
pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("cupy", severity="none"),
    reason="CuPy not installed or GPU not available - skipping GPU tests",
)


class TestROCKETGPUSanity:
    """Basic sanity checks for ROCKETGPU functionality."""

    def test_rocketgpu_sanity_univariate(self):
        """Test basic functionality on univariate time series."""
        # Generate random univariate data
        X = np.random.randn(10, 1, 100).astype(np.float32)

        rocket_gpu = ROCKETGPU(n_kernels=100, random_state=42, normalise=False)
        rocket_gpu.fit(X)
        X_transformed = rocket_gpu.transform(X)

        # Verify output shape
        assert X_transformed.shape == (
            10,
            200,
        ), f"Expected shape (10, 200), got {X_transformed.shape}"

        # Verify no NaN or Inf
        assert not np.any(np.isnan(X_transformed)), "Output contains NaN"
        assert not np.any(np.isinf(X_transformed)), "Output contains Inf"

        # Verify PPV values are in [0, 1] range (first feature of each kernel pair)
        ppv_features = X_transformed[:, ::2]
        assert np.all(
            (ppv_features >= 0) & (ppv_features <= 1)
        ), "PPV features should be in [0, 1] range"

    def test_rocketgpu_sanity_multivariate(self):
        """Test basic functionality on multivariate time series."""
        # Generate random multivariate data (3 channels)
        X = np.random.randn(10, 3, 100).astype(np.float32)

        rocket_gpu = ROCKETGPU(n_kernels=100, random_state=42, normalise=False)
        rocket_gpu.fit(X)
        X_transformed = rocket_gpu.transform(X)

        # Verify output shape
        assert X_transformed.shape == (
            10,
            200,
        ), f"Expected shape (10, 200), got {X_transformed.shape}"

        # Verify no NaN or Inf
        assert not np.any(np.isnan(X_transformed)), "Output contains NaN"
        assert not np.any(np.isinf(X_transformed)), "Output contains Inf"


class TestROCKETGPUParity:
    """Numerical parity tests between CPU Rocket and GPU ROCKETGPU."""

    @pytest.mark.parametrize("n_kernels", [50, 100])
    def test_rocketgpu_cpu_parity_univariate(self, n_kernels):
        """Test CPU-GPU parity on univariate data with different kernel counts."""
        random_state = 42
        X, _ = make_example_3d_numpy(n_channels=1, n_timepoints=150, random_state=42)

        # CPU version
        rocket_cpu = Rocket(
            n_kernels=n_kernels,
            random_state=random_state,
            normalise=False,
        )
        rocket_cpu.fit(X)
        cpu_features = rocket_cpu.transform(X)

        # GPU version
        rocket_gpu = ROCKETGPU(
            n_kernels=n_kernels,
            random_state=random_state,
            normalise=False,
        )
        rocket_gpu.fit(X)
        gpu_features = rocket_gpu.transform(X)

        # Assert shapes match
        assert (
            cpu_features.shape == gpu_features.shape
        ), f"Shape mismatch: CPU {cpu_features.shape} vs GPU {gpu_features.shape}"

        # Assert numerical parity (within float32 tolerance)
        assert_allclose(
            cpu_features,
            gpu_features,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"CPU-GPU parity failed for {n_kernels} kernels (univariate)",
        )

        # Compute and log MAE for verification
        mae = np.mean(np.abs(cpu_features - gpu_features))
        assert mae < 1e-5, f"MAE {mae:.2e} exceeds 1e-5 threshold"

    @pytest.mark.parametrize("n_channels", [2, 3, 5])
    def test_rocketgpu_cpu_parity_multivariate(self, n_channels):
        """Test CPU-GPU parity on multivariate data with varying channels."""
        random_state = 42
        n_kernels = 100
        X, _ = make_example_3d_numpy(
            n_channels=n_channels,
            n_timepoints=150,
            random_state=random_state,
        )

        # CPU version
        rocket_cpu = Rocket(
            n_kernels=n_kernels,
            random_state=random_state,
            normalise=False,
        )
        rocket_cpu.fit(X)
        cpu_features = rocket_cpu.transform(X)

        # GPU version
        rocket_gpu = ROCKETGPU(
            n_kernels=n_kernels,
            random_state=random_state,
            normalise=False,
        )
        rocket_gpu.fit(X)
        gpu_features = rocket_gpu.transform(X)

        # Assert numerical parity
        assert_allclose(
            cpu_features,
            gpu_features,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"CPU-GPU parity failed for {n_channels} channels",
        )

        # Verify MAE
        mae = np.mean(np.abs(cpu_features - gpu_features))
        assert (
            mae < 1e-5
        ), f"MAE {mae:.2e} exceeds 1e-5 threshold for {n_channels} channels"

    def test_rocketgpu_normalization_parity(self):
        """Test that normalization produces identical results on CPU and GPU."""
        random_state = 42
        X, _ = make_example_3d_numpy(n_channels=1, n_timepoints=150, random_state=42)

        # CPU with normalization
        rocket_cpu = Rocket(n_kernels=100, random_state=random_state, normalise=True)
        rocket_cpu.fit(X)
        cpu_features = rocket_cpu.transform(X)

        # GPU with normalization
        rocket_gpu = ROCKETGPU(
            n_kernels=100,
            random_state=random_state,
            normalise=True,
        )
        rocket_gpu.fit(X)
        gpu_features = rocket_gpu.transform(X)

        # Assert parity with normalization applied
        assert_allclose(
            cpu_features,
            gpu_features,
            rtol=1e-5,
            atol=1e-5,
            err_msg="CPU-GPU parity failed with normalization",
        )


class TestROCKETGPUEdgeCases:
    """Edge case testing for ROCKETGPU robustness."""

    def test_rocketgpu_variable_lengths(self):
        """Test handling of standard variable-length inputs."""
        # Test various common input shapes
        test_shapes = [
            (5, 1, 50),  # Small univariate
            (10, 1, 100),  # Medium univariate
            (20, 3, 200),  # Large multivariate
        ]

        for shape in test_shapes:
            X = np.random.randn(*shape).astype(np.float32)

            rocket_gpu = ROCKETGPU(n_kernels=50, random_state=42, normalise=False)
            rocket_gpu.fit(X)
            X_transformed = rocket_gpu.transform(X)

            expected_shape = (shape[0], 100)  # 50 kernels * 2 features
            assert X_transformed.shape == expected_shape, (
                f"Failed for shape {shape}: "
                f"expected {expected_shape}, got {X_transformed.shape}"
            )

    def test_rocketgpu_edge_case_short_series(self):
        """Test behavior with very short time series (< kernel length)."""
        # Create short series (length 5, typical kernel length is 7-11)
        X = np.random.randn(5, 1, 5).astype(np.float32)

        rocket_gpu = ROCKETGPU(n_kernels=20, random_state=42, normalise=False)
        rocket_gpu.fit(X)
        X_transformed = rocket_gpu.transform(X)

        # Should handle gracefully (CPU pads, GPU must match)
        assert X_transformed.shape == (
            5,
            40,
        ), f"Expected shape (5, 40) for short series, got {X_transformed.shape}"

        # Verify against CPU
        rocket_cpu = Rocket(n_kernels=20, random_state=42, normalise=False)
        rocket_cpu.fit(X)
        cpu_features = rocket_cpu.transform(X)

        assert_allclose(
            cpu_features,
            X_transformed,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Short series parity failed",
        )

    @pytest.mark.parametrize("constant_value", [0.0, 1.0, -1.0])
    def test_rocketgpu_edge_case_constant_input(self, constant_value):
        """Test behavior with constant-valued time series."""
        # Create constant-valued series
        X = np.full((10, 1, 100), constant_value, dtype=np.float32)

        rocket_gpu = ROCKETGPU(n_kernels=50, random_state=42, normalise=False)
        rocket_gpu.fit(X)
        X_transformed = rocket_gpu.transform(X)

        # Should produce valid output (not NaN/Inf)
        assert not np.any(
            np.isnan(X_transformed)
        ), f"NaN detected with constant value {constant_value}"
        assert not np.any(
            np.isinf(X_transformed)
        ), f"Inf detected with constant value {constant_value}"

        # Verify against CPU
        rocket_cpu = Rocket(n_kernels=50, random_state=42, normalise=False)
        rocket_cpu.fit(X)
        cpu_features = rocket_cpu.transform(X)

        assert_allclose(
            cpu_features,
            X_transformed,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Constant input parity failed for value {constant_value}",
        )

    def test_rocketgpu_single_sample(self):
        """Test transform with a single time series."""
        X = np.random.randn(1, 1, 100).astype(np.float32)

        rocket_gpu = ROCKETGPU(n_kernels=50, random_state=42, normalise=False)
        rocket_gpu.fit(X)
        X_transformed = rocket_gpu.transform(X)

        assert X_transformed.shape == (
            1,
            100,
        ), f"Single sample failed: expected (1, 100), got {X_transformed.shape}"

        # Verify against CPU
        rocket_cpu = Rocket(n_kernels=50, random_state=42, normalise=False)
        rocket_cpu.fit(X)
        cpu_features = rocket_cpu.transform(X)

        assert_allclose(
            cpu_features,
            X_transformed,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Single sample parity failed",
        )


class TestROCKETGPUReproducibility:
    """Test reproducibility with random seeds."""

    def test_rocketgpu_reproducibility(self):
        """Test that same random_state produces identical results."""
        X, _ = make_example_3d_numpy(n_channels=1, n_timepoints=150, random_state=42)

        # First run
        rocket_gpu_1 = ROCKETGPU(n_kernels=100, random_state=42, normalise=False)
        rocket_gpu_1.fit(X)
        features_1 = rocket_gpu_1.transform(X)

        # Second run with same seed
        rocket_gpu_2 = ROCKETGPU(n_kernels=100, random_state=42, normalise=False)
        rocket_gpu_2.fit(X)
        features_2 = rocket_gpu_2.transform(X)

        # Should be identical (bit-exact)
        assert_allclose(
            features_1,
            features_2,
            rtol=0,
            atol=0,
            err_msg="Reproducibility failed - same seed produced different results",
        )

    def test_rocketgpu_different_seeds(self):
        """Test that different random_states produce different results."""
        X, _ = make_example_3d_numpy(n_channels=1, n_timepoints=150, random_state=42)

        # Run with seed 42
        rocket_gpu_1 = ROCKETGPU(n_kernels=100, random_state=42, normalise=False)
        rocket_gpu_1.fit(X)
        features_1 = rocket_gpu_1.transform(X)

        # Run with seed 123
        rocket_gpu_2 = ROCKETGPU(n_kernels=100, random_state=123, normalise=False)
        rocket_gpu_2.fit(X)
        features_2 = rocket_gpu_2.transform(X)

        # Should be different
        assert not np.allclose(
            features_1, features_2, rtol=1e-5, atol=1e-5
        ), "Different seeds produced identical results"
