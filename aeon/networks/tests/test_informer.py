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
    "seq_len,label_len,out_len,d_model,n_heads,e_layers,d_layers",
    [
        (96, 48, 24, 512, 8, 3, 2),
        (48, 24, 12, 256, 4, 2, 1),
        (120, 60, 30, 128, 2, 1, 1),
        (72, 36, 18, 64, 1, 2, 2),
    ],
)
def test_informer_network_init(
    seq_len,
    label_len,
    out_len,
    d_model,
    n_heads,
    e_layers,
    d_layers,
):
    """Test whether InformerNetwork initializes correctly for various parameters."""
    informer = InformerNetwork(
        seq_len=seq_len,
        label_len=label_len,
        out_len=out_len,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        factor=random.choice([3, 5, 7]),
        dropout=random.choice([0.0, 0.1, 0.2]),
        attn=random.choice(["prob", "full"]),
        activation=random.choice(["relu", "gelu"]),
    )

    inputs, outputs = informer.build_network((seq_len + label_len, 5))
    assert inputs is not None
    assert outputs is not None
    assert len(inputs) == 2  # encoder_input and decoder_input


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "attn,activation",
    [("prob", "relu"), ("full", "gelu"), ("prob", "gelu"), ("full", "relu")],
)
def test_informer_network_attention_activation(attn, activation):
    """Test InformerNetwork with different attention and activation."""
    informer = InformerNetwork(
        seq_len=96,
        label_len=48,
        out_len=24,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        attn=attn,
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
        seq_len=48,
        label_len=24,
        out_len=12,
        d_model=64,
        n_heads=2,
        e_layers=1,
        d_layers=1,
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
def test_informer_network_output_shape():
    """Test whether InformerNetwork produces correct output shapes."""
    seq_len = 96
    label_len = 48
    out_len = 24
    n_channels = 5
    # batch_size = 32

    informer = InformerNetwork(
        seq_len=seq_len,
        label_len=label_len,
        out_len=out_len,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
    )

    inputs, outputs = informer.build_network((seq_len + label_len, n_channels))

    # Create a TensorFlow model to test actual shapes
    if _check_soft_dependencies(["tensorflow"], severity="none"):
        # keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Test input shapes
        encoder_input_shape = inputs[0].shape
        decoder_input_shape = inputs[1].shape

        assert encoder_input_shape[1] == seq_len  # sequence length
        assert encoder_input_shape[2] == n_channels  # number of channels
        assert decoder_input_shape[1] == label_len + out_len  # decoder sequence length
        assert decoder_input_shape[2] == n_channels  # number of channels

        # Test output shape
        output_shape = outputs.shape
        assert output_shape[1] == out_len  # prediction length
        assert output_shape[2] == n_channels  # number of channels


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
    assert informer.seq_len == 96
    assert informer.label_len == 48
    assert informer.out_len == 24
    assert informer.d_model == 512
    assert informer.n_heads == 8
    assert informer.e_layers == 3
    assert informer.d_layers == 2
    assert informer.attn == "prob"
    assert informer.activation == "gelu"
    assert informer.distil
    assert informer.mix


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
def test_informer_network_parameter_validation():
    """Test whether InformerNetwork handles edge case parameters correctly."""
    # Test minimum viable configuration
    informer = InformerNetwork(
        seq_len=12,
        label_len=6,
        out_len=3,
        d_model=32,
        n_heads=1,
        e_layers=1,
        d_layers=1,
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
            seq_len=48,
            label_len=24,
            out_len=12,
            d_model=64,
            n_heads=2,
            e_layers=1,
            d_layers=1,
        )

        inputs, outputs = informer.build_network((72, n_channels))
        assert inputs is not None
        assert outputs is not None
