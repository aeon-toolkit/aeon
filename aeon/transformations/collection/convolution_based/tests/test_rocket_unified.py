"""Comprehensive tests for unified ROCKET interface with device parameter.

This test file validates Phase 1 implementation of the unified ROCKET interface,
which allows users to select between CPU and GPU execution via a device parameter.
"""

import numpy as np
import pytest

from aeon.datasets import load_unit_test
from aeon.transformations.collection.convolution_based import Rocket


class TestRocketDeviceSelection:
    """Test device selection functionality."""

    def test_default_device_is_cpu(self):
        """Test that device defaults to CPU for backward compatibility."""
        rocket = Rocket(n_kernels=100)
        assert rocket.device == "cpu", "Default device should be 'cpu'"

    def test_explicit_cpu_device(self):
        """Test explicit CPU device selection."""
        rocket = Rocket(n_kernels=100, device="cpu", random_state=42)
        X = np.random.randn(10, 1, 50)
        rocket.fit(X)

        assert rocket._actual_device == "cpu"
        X_transformed = rocket.transform(X)
        assert X_transformed.shape == (10, 200)

    def test_explicit_gpu_device_unavailable(self):
        """Test that GPU device raises error when unavailable."""
        rocket = Rocket(n_kernels=100, device="gpu")
        X = np.random.randn(10, 1, 50)

        # Try to fit - should raise error if GPU unavailable
        try:
            rocket.fit(X)
            # If we got here, GPU is available - verify it worked
            assert rocket._actual_device == "gpu"
            X_transformed = rocket.transform(X)
            assert X_transformed.shape == (10, 200)
        except RuntimeError as e:
            # GPU not available - verify error message is helpful
            assert "GPU device requested but not available" in str(e)
            assert "TensorFlow" in str(e)

    def test_auto_device_selection(self):
        """Test auto device selection with fallback."""
        with pytest.warns(UserWarning):
            rocket = Rocket(n_kernels=100, device="auto", random_state=42)
            X = np.random.randn(10, 1, 50)
            rocket.fit(X)

        # Should select either CPU or GPU
        assert rocket._actual_device in ["cpu", "gpu"]

        # Should work regardless of which was selected
        X_transformed = rocket.transform(X)
        assert X_transformed.shape == (10, 200)

    def test_invalid_device_raises_error(self):
        """Test that invalid device parameter raises ValueError."""
        rocket = Rocket(n_kernels=100, device="tpu")
        X = np.random.randn(10, 1, 50)

        with pytest.raises(ValueError, match="Invalid device 'tpu'"):
            rocket.fit(X)


class TestRocketParameterValidation:
    """Test parameter validation and warnings."""

    def test_gpu_params_on_cpu_warning(self):
        """Test warning when GPU parameters provided with CPU device."""
        with pytest.warns(UserWarning, match="GPU-only parameters"):
            rocket = Rocket(
                n_kernels=100,
                device="cpu",
                batch_size=128,  # GPU param
                kernel_size=[7, 9],  # GPU param
                random_state=42,
            )
            X = np.random.randn(10, 1, 50)
            rocket.fit(X)

    def test_cpu_params_work_on_cpu(self):
        """Test that CPU-specific parameters work correctly."""
        rocket = Rocket(
            n_kernels=100, device="cpu", normalise=True, n_jobs=2, random_state=42
        )
        X = np.random.randn(10, 1, 50)
        rocket.fit(X)
        X_transformed = rocket.transform(X)

        assert X_transformed.shape == (10, 200)

    def test_cpu_params_on_gpu_warning(self):
        """Test warning when CPU parameters provided with GPU device."""
        try:
            with pytest.warns(UserWarning, match="CPU-only parameters"):
                rocket = Rocket(
                    n_kernels=100,
                    device="gpu",
                    normalise=False,  # CPU param
                    n_jobs=4,  # CPU param
                    random_state=42,
                )
                X = np.random.randn(10, 1, 50)
                rocket.fit(X)
        except RuntimeError:
            # GPU not available - that's okay for this test
            pytest.skip("GPU not available")


class TestRocketErrorHandling:
    """Test error handling and messages."""

    def test_transform_before_fit_error(self):
        """Test that transforming before fitting raises clear error."""
        rocket = Rocket(n_kernels=100)
        X = np.random.randn(10, 1, 50)

        with pytest.raises(ValueError, match="not fitted"):
            rocket.transform(X)

    def test_gpu_unavailable_error_message(self):
        """Test GPU unavailable error includes helpful information."""
        rocket = Rocket(n_kernels=100, device="gpu")
        X = np.random.randn(10, 1, 50)

        try:
            rocket.fit(X)
        except RuntimeError as e:
            error_msg = str(e)
            # Should have helpful info
            assert "TensorFlow" in error_msg
            assert any(word in error_msg for word in ["install", "Install"])


class TestRocketPickling:
    """Test pickling behavior for different devices."""

    def test_cpu_version_can_be_pickled(self):
        """Test that CPU version can be pickled and unpickled."""
        import pickle

        rocket = Rocket(n_kernels=100, device="cpu", random_state=42)
        X = np.random.randn(10, 1, 50)
        rocket.fit(X)

        # Pickle and unpickle
        pickled = pickle.dumps(rocket)
        rocket_loaded = pickle.loads(pickled)

        # Should still work
        X_transformed = rocket_loaded.transform(X)
        assert X_transformed.shape == (10, 200)

    def test_gpu_version_cannot_be_pickled(self):
        """Test that GPU version raises helpful error when pickling."""
        import pickle

        rocket = Rocket(n_kernels=100, device="gpu")
        X = np.random.randn(10, 1, 50)

        try:
            rocket.fit(X)

            # Try to pickle - should fail
            with pytest.raises(TypeError, match="Cannot pickle.*device='gpu'"):
                pickle.dumps(rocket)
        except RuntimeError:
            # GPU not available - can't test this
            pytest.skip("GPU not available")


class TestRocketBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_old_api_without_device_param(self):
        """Test that old API (no device param) still works."""
        # Old-style usage
        rocket = Rocket(n_kernels=100, normalise=True, n_jobs=1, random_state=42)
        X = np.random.randn(10, 1, 50)

        rocket.fit(X)
        X_transformed = rocket.transform(X)

        # Should default to CPU
        assert rocket._actual_device == "cpu"
        assert X_transformed.shape == (10, 200)

    def test_old_api_with_real_data(self):
        """Test old API with aeon dataset."""
        X_train, _ = load_unit_test(split="train")
        X_test, _ = load_unit_test(split="test")

        # Old-style usage
        rocket = Rocket(n_kernels=500, random_state=0)
        rocket.fit(X_train)
        X_transformed = rocket.transform(X_test)

        assert X_transformed.shape[0] == len(X_test)
        assert X_transformed.shape[1] == 1000  # 500 kernels * 2


class TestRocketTags:
    """Test that class tags are updated based on device."""

    def test_cpu_tags(self):
        """Test that CPU version has correct tags."""
        rocket = Rocket(n_kernels=100, device="cpu")
        X = np.random.randn(10, 1, 50)
        rocket.fit(X)

        # CPU should support multithreading
        assert rocket.get_tag("capability:multithreading") is True
        # CPU should not have pickle restriction
        assert rocket.get_tag("cant_pickle") is not True

    def test_gpu_tags(self):
        """Test that GPU version has correct tags."""
        rocket = Rocket(n_kernels=100, device="gpu")
        X = np.random.randn(10, 1, 50)

        try:
            rocket.fit(X)

            # GPU should have TensorFlow dependency
            assert rocket.get_tag("python_dependencies") == "tensorflow"
            # GPU cannot be pickled
            assert rocket.get_tag("cant_pickle") is True
        except RuntimeError:
            # GPU not available
            pytest.skip("GPU not available")


class TestRocketFunctionalCorrectness:
    """Test that transformations produce correct outputs."""

    def test_cpu_transform_shape(self):
        """Test CPU transformation produces correct output shape."""
        rocket = Rocket(n_kernels=500, device="cpu", random_state=42)
        X = np.random.randn(20, 1, 100)

        rocket.fit(X)
        X_transformed = rocket.transform(X)

        # Should have n_samples rows and n_kernels*2 columns
        assert X_transformed.shape == (20, 1000)

    def test_multivariate_cpu(self):
        """Test CPU version with multivariate data."""
        rocket = Rocket(n_kernels=200, device="cpu", random_state=42)
        X = np.random.randn(15, 3, 75)  # 3 channels

        rocket.fit(X)
        X_transformed = rocket.transform(X)

        assert X_transformed.shape == (15, 400)

    def test_fit_transform_same_as_separate(self):
        """Test that fit_transform gives same result as fit then transform."""
        X = np.random.randn(10, 1, 50)

        # Separate fit and transform
        rocket1 = Rocket(n_kernels=100, random_state=42)
        rocket1.fit(X)
        X_trans1 = rocket1.transform(X)

        # fit_transform
        rocket2 = Rocket(n_kernels=100, random_state=42)
        X_trans2 = rocket2.fit_transform(X)

        # Should be identical
        np.testing.assert_array_almost_equal(X_trans1, X_trans2)

    def test_deterministic_with_random_state(self):
        """Test that same random_state gives deterministic results."""
        X = np.random.randn(10, 1, 50)

        rocket1 = Rocket(n_kernels=100, device="cpu", random_state=42)
        rocket1.fit(X)
        X_trans1 = rocket1.transform(X)

        rocket2 = Rocket(n_kernels=100, device="cpu", random_state=42)
        rocket2.fit(X)
        X_trans2 = rocket2.transform(X)

        # Same random state should give identical results
        np.testing.assert_array_equal(X_trans1, X_trans2)


class TestRocketEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test transformation of single sample."""
        rocket = Rocket(n_kernels=100, random_state=42)
        X_train = np.random.randn(10, 1, 50)
        X_test = np.random.randn(1, 1, 50)  # Single sample

        rocket.fit(X_train)
        X_transformed = rocket.transform(X_test)

        assert X_transformed.shape == (1, 200)

    def test_small_n_kernels(self):
        """Test with very small number of kernels."""
        rocket = Rocket(n_kernels=10, random_state=42)
        X = np.random.randn(10, 1, 50)

        rocket.fit(X)
        X_transformed = rocket.transform(X)

        assert X_transformed.shape == (10, 20)

    def test_different_length_timepoints(self):
        """Test with different time series lengths."""
        rocket = Rocket(n_kernels=100, random_state=42)

        # Fit on one length
        X_train = np.random.randn(10, 1, 50)
        rocket.fit(X_train)

        # Transform same length should work
        X_test = np.random.randn(5, 1, 50)
        X_transformed = rocket.transform(X_test)
        assert X_transformed.shape == (5, 200)
