"""Hydra tests."""

import numpy as np
import pytest

from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_output_types():
    """Test HydraTransformer output_type parameter."""
    # Create a simple dataset
    X = np.random.random(size=(10, 3, 20))

    # Test tensor output (default)
    hydra_tensor = HydraTransformer(random_state=42)
    hydra_tensor.fit(X)
    tensor_output = hydra_tensor.transform(X)

    # Check that output is a torch tensor
    import torch

    assert isinstance(tensor_output, torch.Tensor)

    # Test numpy output
    hydra_numpy = HydraTransformer(random_state=42, output_type="numpy")
    hydra_numpy.fit(X)
    numpy_output = hydra_numpy.transform(X)

    # Check that output is a numpy array
    assert isinstance(numpy_output, np.ndarray)

    # Test dataframe output
    hydra_df = HydraTransformer(random_state=42, output_type="dataframe")
    hydra_df.fit(X)
    df_output = hydra_df.transform(X)

    # Check that output is a pandas DataFrame
    import pandas as pd

    assert isinstance(df_output, pd.DataFrame)

    # Check that all outputs have the same shape
    assert tensor_output.shape == numpy_output.shape
    assert numpy_output.shape == df_output.shape

    tensor_np = tensor_output.detach().cpu().numpy()
    df_np = df_output.to_numpy()
    assert np.allclose(tensor_np[:5], numpy_output[:5], atol=1e-6)
    assert np.allclose(numpy_output[:5], df_np[:5], atol=1e-6)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_short_series():
    """Test HydraTransformer with very short time series."""
    # Create a dataset with short time series (less than kernel length)
    X = np.random.random(size=(10, 2, 8))

    hydra = HydraTransformer(random_state=42)
    # Should still work with short series, but will use padding
    hydra.fit(X)
    output = hydra.transform(X)

    # Check that output has the expected number of samples
    assert output.shape[0] == 10


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_parameter_values():
    """Test HydraTransformer with different parameter values."""
    X = np.random.random(size=(10, 3, 20))

    # Test with different numbers of kernels and groups
    hydra = HydraTransformer(n_kernels=4, n_groups=32, random_state=42)
    hydra.fit(X)
    output = hydra.transform(X)

    # Expected feature count: num_dilations * divisor * h * k
    # where divisor = min(2, g), h = g // divisor
    # For a 20-length time series, num_dilations should be around 2-3
    # With n_groups=32, divisor=2, h=16, n_kernels=4
    # So expected features would be approximately 2 * 2 * 16 * 4 = 256
    # But this is approximate since dilations depend on series length
    assert output.shape[1] > 0


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_univariate():
    """Test HydraTransformer with univariate data."""
    # Create a univariate dataset
    X = np.random.random(size=(10, 1, 20))

    hydra = HydraTransformer(random_state=42)
    hydra.fit(X)
    output = hydra.transform(X)

    # Check that output has the expected number of samples
    assert output.shape[0] == 10
    assert output.shape[1] > 0
