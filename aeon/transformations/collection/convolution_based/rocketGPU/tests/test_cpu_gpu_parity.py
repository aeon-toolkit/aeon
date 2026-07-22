"""Tests for CPU-GPU parity in ROCKET implementations.

This test suite verifies that the ROCKETGPU implementation produces results
consistent with the CPU ROCKET when initialized with the same random seed.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.convolution_based._rocket import Rocket
from aeon.transformations.collection.convolution_based.rocketGPU._rocket_gpu import (
    ROCKETGPU,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("n_channels", [1, 2, 3, 8])
@pytest.mark.parametrize("n_timepoints", [50, 100])
def test_cpu_gpu_parity_basic(n_channels, n_timepoints):
    """Test that CPU and GPU produce consistent features for various configurations.

    Covers univariate and multivariate cases with different channel counts
    and series lengths.
    """
    random_state = 42
    n_cases = 20
    n_kernels = 50  # Smaller for faster tests

    X, _ = make_example_3d_numpy(
        n_cases=n_cases,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        random_state=random_state,
    )

    cpu = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    gpu = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    cpu.fit(X)
    gpu.fit(X)

    X_cpu = cpu.transform(X)
    X_gpu = gpu.transform(X)

    # Features should match within floating-point tolerance
    assert_array_almost_equal(X_cpu, X_gpu, decimal=4)

    # Verify shapes are identical
    assert X_cpu.shape == X_gpu.shape
    assert X_cpu.shape == (n_cases, n_kernels * 2)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_cpu_gpu_determinism():
    """Test that same seed produces identical results across multiple runs."""
    random_state = 123
    n_kernels = 50

    X, _ = make_example_3d_numpy(n_cases=15, n_channels=3, n_timepoints=75)

    # First run
    cpu1 = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    gpu1 = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    X_cpu1 = cpu1.fit_transform(X)
    X_gpu1 = gpu1.fit_transform(X)

    # Second run with same seed
    cpu2 = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    gpu2 = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    X_cpu2 = cpu2.fit_transform(X)
    X_gpu2 = gpu2.fit_transform(X)

    # CPU should be deterministic
    assert_array_almost_equal(X_cpu1, X_cpu2, decimal=10)

    # GPU should be deterministic
    assert_array_almost_equal(X_gpu1, X_gpu2, decimal=10)

    # CPU and GPU should match each other
    assert_array_almost_equal(X_cpu1, X_gpu1, decimal=4)
    assert_array_almost_equal(X_cpu2, X_gpu2, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("seed1,seed2", [(42, 100), (0, 999), (123, 456)])
def test_cpu_gpu_different_seeds_produce_different_results(seed1, seed2):
    """Test that different seeds produce different transformations."""
    X, _ = make_example_3d_numpy(n_cases=10, n_channels=2, n_timepoints=60)
    n_kernels = 30

    # First seed
    cpu1 = Rocket(n_kernels=n_kernels, random_state=seed1, normalise=False)
    gpu1 = ROCKETGPU(n_kernels=n_kernels, random_state=seed1)
    X_cpu1 = cpu1.fit_transform(X)
    X_gpu1 = gpu1.fit_transform(X)

    # Different seed
    cpu2 = Rocket(n_kernels=n_kernels, random_state=seed2, normalise=False)
    gpu2 = ROCKETGPU(n_kernels=n_kernels, random_state=seed2)
    X_cpu2 = cpu2.fit_transform(X)
    X_gpu2 = gpu2.fit_transform(X)

    # Different seeds should produce different results
    # Using a simple check: not all values should be close
    max_diff_cpu = np.abs(X_cpu1 - X_cpu2).max()
    max_diff_gpu = np.abs(X_gpu1 - X_gpu2).max()

    # At least some difference should exist (not bit-identical)
    assert max_diff_cpu > 1e-6, "CPU results are too similar for different seeds"
    assert max_diff_gpu > 1e-6, "GPU results are too similar for different seeds"

    # But for same seed, CPU and GPU should still match
    assert_array_almost_equal(X_cpu1, X_gpu1, decimal=4)
    assert_array_almost_equal(X_cpu2, X_gpu2, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_cpu_gpu_parity_minimum_data():
    """Test parity with minimal dataset size.

    Edge case: smallest reasonable dataset to ensure there are no
    boundary issues with small data.
    """
    random_state = 42
    n_kernels = 20

    # Minimal reasonable sizes
    X, _ = make_example_3d_numpy(n_cases=5, n_channels=1, n_timepoints=30)

    cpu = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    gpu = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    X_cpu = cpu.fit_transform(X)
    X_gpu = gpu.fit_transform(X)

    assert_array_almost_equal(X_cpu, X_gpu, decimal=4)


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_cpu_gpu_parity_long_series():
    """Test parity with longer time series.

    Verifies that accumulated floating-point errors don't grow
    significantly with longer convolution operations.
    """
    random_state = 42
    n_kernels = 50

    # Longer series to test accumulation behavior
    X, _ = make_example_3d_numpy(n_cases=10, n_channels=4, n_timepoints=200)

    cpu = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    gpu = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    X_cpu = cpu.fit_transform(X)
    X_gpu = gpu.fit_transform(X)

    assert_array_almost_equal(X_cpu, X_gpu, decimal=4)

    # Additional check: correlation should be near-perfect
    correlation = np.corrcoef(X_cpu.flatten(), X_gpu.flatten())[0, 1]
    assert correlation > 0.9999, f"Correlation {correlation} is lower than expected"


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_cpu_gpu_fit_transform_consistency():
    """Test that fit_transform produces same results as fit then transform."""
    random_state = 42
    n_kernels = 40

    X, _ = make_example_3d_numpy(n_cases=15, n_channels=3, n_timepoints=80)

    # Separate fit and transform
    cpu_sep = Rocket(n_kernels=n_kernels, random_state=random_state, normalise=False)
    gpu_sep = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    cpu_sep.fit(X)
    gpu_sep.fit(X)
    X_cpu_sep = cpu_sep.transform(X)
    X_gpu_sep = gpu_sep.transform(X)

    # Combined fit_transform
    cpu_combined = Rocket(
        n_kernels=n_kernels, random_state=random_state, normalise=False
    )
    gpu_combined = ROCKETGPU(n_kernels=n_kernels, random_state=random_state)

    X_cpu_combined = cpu_combined.fit_transform(X)
    X_gpu_combined = gpu_combined.fit_transform(X)

    # Both methods should produce identical results
    assert_array_almost_equal(X_cpu_sep, X_cpu_combined, decimal=10)
    assert_array_almost_equal(X_gpu_sep, X_gpu_combined, decimal=10)

    # And CPU-GPU parity should hold
    assert_array_almost_equal(X_cpu_combined, X_gpu_combined, decimal=4)
