"""Tests for the Informer Network Model."""

import random

import pytest

from aeon.networks import InformerNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "encoder_input_len,decoder_input_len,"
    "prediction_horizon,model_dimension,num_attention_heads,"
    "encoder_layers,decoder_layers",
    [
        (96, 48, 24, 512, 8, 3, 2),
        (48, 24, 12, 256, 4, 2, 1),
        (120, 60, 30, 128, 2, 1, 1),
        (72, 36, 18, 64, 1, 2, 2),
    ],
)
def test_informer_network_init(
    encoder_input_len,
    decoder_input_len,
    prediction_horizon,
    model_dimension,
    num_attention_heads,
    encoder_layers,
    decoder_layers,
):
    """Test whether InformerNetwork initializes correctly for various parameters."""
    informer = InformerNetwork(
        encoder_input_len=encoder_input_len,
        decoder_input_len=decoder_input_len,
        prediction_horizon=prediction_horizon,
        model_dimension=model_dimension,
        num_attention_heads=num_attention_heads,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        factor=random.choice([3, 5, 7]),
        dropout=random.choice([0.0, 0.1, 0.2]),
        attention_type=random.choice(["prob", "full"]),
        activation=random.choice(["relu", "gelu"]),
    )

    inputs, outputs = informer.build_network((encoder_input_len + decoder_input_len, 5))
    assert inputs is not None
    assert outputs is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "attention_type,activation",
    [("prob", "relu"), ("full", "gelu"), ("prob", "gelu"), ("full", "relu")],
)
def test_informer_network_attention_activation(attention_type, activation):
    """Test InformerNetwork with different attention and activation."""
    informer = InformerNetwork(
        encoder_input_len=96,
        decoder_input_len=48,
        prediction_horizon=24,
        model_dimension=128,
        num_attention_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        attention_type=attention_type,
        activation=activation,
    )

    inputs, outputs = informer.build_network((144, 3))
    assert inputs is not None
    assert outputs is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "distil,mix,factor",
    [(True, True, 5), (False, False, 3), (True, False, 7), (False, True, 2)],
)
def test_informer_network_distil_mix_factor(distil, mix, factor):
    """Test whether InformerNetwork works with different configurations."""
    informer = InformerNetwork(
        encoder_input_len=48,
        decoder_input_len=24,
        prediction_horizon=12,
        model_dimension=64,
        num_attention_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        distil=distil,
        mix=mix,
        factor=factor,
    )

    inputs, outputs = informer.build_network((72, 2))
    assert inputs is not None
    assert outputs is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_informer_network_default_parameters():
    """Test whether InformerNetwork works with default parameters."""
    informer = InformerNetwork()

    inputs, outputs = informer.build_network((120, 1))
    assert inputs is not None
    assert outputs is not None

    # Check default values
    assert informer.encoder_input_len == 96
    assert informer.decoder_input_len == 48
    assert informer.prediction_horizon == 24
    assert informer.model_dimension == 512
    assert informer.num_attention_heads == 8
    assert informer.encoder_layers == 3
    assert informer.decoder_layers == 2
    assert informer.attention_type == "prob"
    assert informer.activation == "gelu"
    assert informer.distil
    assert informer.mix


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_informer_network_parameter_validation():
    """Test whether InformerNetwork handles edge case parameters correctly."""
    informer = InformerNetwork(
        encoder_input_len=12,
        decoder_input_len=6,
        prediction_horizon=3,
        model_dimension=32,
        num_attention_heads=1,
        encoder_layers=1,
        decoder_layers=1,
        factor=1,
        dropout=0.0,
    )

    inputs, outputs = informer.build_network((18, 1))
    assert inputs is not None
    assert outputs is not None


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_informer_network_different_channels():
    """Test whether InformerNetwork works with different numbers of input channels."""
    for n_channels in [1, 3, 5, 10]:
        informer = InformerNetwork(
            encoder_input_len=48,
            decoder_input_len=24,
            prediction_horizon=12,
            model_dimension=64,
            num_attention_heads=2,
            encoder_layers=1,
            decoder_layers=1,
        )

        inputs, outputs = informer.build_network((72, n_channels))
        assert inputs is not None
        assert outputs is not None
