"""Tests for the AutoEncoder Networks."""

import pytest

from aeon.networks import (
    AEDCNNNetwork,
    AEDRNNNetwork,
    AEFCNNetwork,
    AERecurrentNetwork,
    AEResNetNetwork,
)
from aeon.typing import LATENT_SPACE
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "Model",
    [AEDCNNNetwork, AEDRNNNetwork, AEFCNNetwork, AEResNetNetwork, AERecurrentNetwork],
)
@pytest.mark.parametrize(
    "latent_space_type",
    [
        LATENT_SPACE.FLAT,
        LATENT_SPACE.TIME,
        LATENT_SPACE.REPEATED,
        "FlAt",
        "TiMe",
        "RePeAteD",
        "unknown",  # This should raise a ValueError
    ],
)
def test_ae_latent_space(Model, latent_space_type):
    """Test for temporal latent space."""
    import tensorflow as tf

    model = Model(latent_space_dim=64, latent_space_type=latent_space_type)

    if latent_space_type == "unknown":
        with pytest.raises(ValueError):
            model.build_network((1000, 5))
        return

    model.build_network((1000, 5))
    encoder = model.get_encoder_model()
    decoder = model.get_decoder_model()
    assert isinstance(encoder, tf.keras.models.Model)
    assert isinstance(decoder, tf.keras.models.Model)
    assert encoder.output_shape[-1] == 64
    assert decoder.input_shape[-1] == 64

    if latent_space_type.lower() == LATENT_SPACE.TIME:
        assert len(encoder.output_shape) == 3
        assert len(decoder.input_shape) == 3
    else:
        assert len(encoder.output_shape) == 2
        assert len(decoder.input_shape) == 2

    if latent_space_type == LATENT_SPACE.REPEATED:
        # check decoder use RepeatVector layer
        assert any(
            isinstance(layer, tf.keras.layers.RepeatVector) for layer in decoder.layers
        )
