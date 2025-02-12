"""Tests for the MLPNetwork Model."""

import pytest
from aeon.networks import MLPNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies

@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "n_layers, n_units, activation",
    [
        (3, 500, "relu"),  
        (5, [256, 128, 128, 64, 32], "sigmoid"),  
        (2, 128, ["tanh", "relu"]),  
    ],
)
def test_mlp_initialization(n_layers, n_units, activation):
    """Test whether MLPNetwork initializes correctly with different configurations."""
    mlp = MLPNetwork(n_layers=n_layers, n_units=n_units, activation=activation)
    input_layer, output_layer = mlp.build_network((1000, 5))
    assert input_layer is not None
    assert output_layer is not None

@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dropout_rate, n_layers",
    [
        (0.2, 3),  
        ([0.1, 0.2, 0.3], 3),
        pytest.param([0.1, 0.2], 3, marks=pytest.mark.xfail(raises=AssertionError)),  # Mismatch
    ],
)
def test_dropout_rate(dropout_rate, n_layers):
    """Test MLPNetwork dropout_rate configurations."""
    mlp = MLPNetwork(n_layers=n_layers, dropout_rate=dropout_rate)
    input_layer, output_layer = mlp.build_network((1000, 5))
    assert input_layer is not None
    assert output_layer is not None

@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "dropout_last",
    [
        (0.3),  
        (0.5),  
        pytest.param(1.2, marks=pytest.mark.xfail(raises=AssertionError)),  # Invalid case
    ],
)
def test_dropout_last(dropout_last):
    """Test MLPNetwork dropout_last configurations."""
    mlp = MLPNetwork(dropout_last=dropout_last)
    input_layer, output_layer = mlp.build_network((1000, 5))
    assert input_layer is not None
    assert output_layer is not None

@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "use_bias",
    [
        (True),  
        (False), 
    ],
)
def test_use_bias(use_bias):
    """Test MLPNetwork use_bias configurations."""
    mlp = MLPNetwork(use_bias=use_bias)
    input_layer, output_layer = mlp.build_network((1000, 5))
    assert input_layer is not None
    assert output_layer is not None

