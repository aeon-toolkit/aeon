"""Tests for the AEBiGRU Model."""

import pytest

from aeon.networks import AEDeTSECNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize(
    "enc_filters, dec_filters, enc_act, dec_act",
    [
        (32, 64, "relu", "relu"),
        (64, 32, "tanh", "tanh"),
        (128, 128, "relu", "tanh"),
        (256, 256, "tanh", "relu"),
    ],
)
def test_ae_detsec_network_init(enc_filters, dec_filters, enc_act, dec_act):
    """Test whether AEDeTSECNetwork initializes correctly for various parameters."""
    aebigru = AEDeTSECNetwork(
        n_filters_encoder=enc_filters,
        n_filters_decoder=dec_filters,
        activation_encoder=enc_act,
        activation_decoder=dec_act,
    )
    encoder, decoder = aebigru.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None
