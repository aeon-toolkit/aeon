"""Hydra tests."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

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


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_batch_matches_forward():
    """Batched kernel application equals a single pass over all cases.

    Covers both branches of ``batch``: everything in one batch when
    ``n_cases <= batch_size``, and split-and-concatenate when smaller
    batches are requested.
    """
    import torch

    n_cases = 10
    X = np.random.default_rng(0).random(size=(n_cases, 1, 20))

    hydra = HydraTransformer(n_kernels=4, n_groups=8, random_state=42)
    hydra.fit(X)
    X_tensor = torch.tensor(X).float()

    full = hydra._hydra(X_tensor)

    assert torch.equal(hydra._hydra.batch(X_tensor), full)
    assert torch.equal(hydra._hydra.batch(X_tensor, batch_size=4), full)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_single_group():
    """A single kernel group transforms without the first-difference branch.

    With ``n_groups=1`` the group divisor is 1, so ``forward`` never takes
    its first-difference path; the output keeps the feature-count invariant
    ``2 * n_kernels * n_groups * num_dilations``.
    """
    n_cases, n_kernels, n_groups, n_timepoints = 6, 2, 1, 20
    X = np.random.default_rng(0).random(size=(n_cases, 1, n_timepoints))

    hydra = HydraTransformer(n_kernels=n_kernels, n_groups=n_groups, random_state=42)
    Xt = hydra.fit_transform(X)

    num_dilations = int(np.log2((n_timepoints - 1) / 8)) + 1
    assert Xt.shape == (n_cases, 2 * n_kernels * n_groups * num_dilations)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="Skip test if torch not available",
)
def test_hydra_random_state_instance():
    """Test seeded RandomState instances produce reproducible transforms."""
    X = np.random.default_rng(0).random(size=(10, 2, 20))
    hydra1 = HydraTransformer(
        n_kernels=4,
        n_groups=8,
        random_state=check_random_state(42),
        output_type="numpy",
    )
    hydra2 = HydraTransformer(
        n_kernels=4,
        n_groups=8,
        random_state=check_random_state(42),
        output_type="numpy",
    )

    Xt1 = hydra1.fit_transform(X)
    Xt2 = hydra2.fit_transform(X)

    np.testing.assert_array_equal(Xt1, Xt2)
