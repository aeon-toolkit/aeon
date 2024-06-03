"""Tests for the AEDRNN Model."""

import random

import pytest

from aeon.networks import AEDRNNNetwork
from aeon.utils.validation._dependencies import _check_soft_dependencies


def pytest_generate_tests():
    """Parameter generation for test cases."""
    latent_space_dim_range = [32, 128, 256]
    n_layers_encoder_range = range(1, 5)
    n_layers_decoder_range = range(1, 3)
    temporal_latent_space_options = [True, False]

    test_params = []
    for latent_space_dim in latent_space_dim_range:
        for n_layers_encoder in n_layers_encoder_range:
            for n_layers_decoder in n_layers_decoder_range:
                for temporal_latent_space in temporal_latent_space_options:
                    test_params.append(
                        dict(
                            latent_space_dim=latent_space_dim,
                            n_layers_encoder=n_layers_encoder,
                            n_layers_decoder=n_layers_decoder,
                            dilation_rate=None,
                            activation_encoder=random.choice(["relu", "tanh"]),
                            activation_decoder=random.choice(["relu", "tanh"]),
                            n_units=[
                                random.choice([50, 25, 100])
                                for _ in range(n_layers_encoder)
                            ],
                            temporal_latent_space=temporal_latent_space,
                        )
                    )
    return test_params


params = pytest_generate_tests()


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow"], severity="none"),
    reason="Tensorflow soft dependency unavailable.",
)
@pytest.mark.parametrize("params", params)
def test_aedrnnnetwork_init(params):
    """Test whether AEDRNNNetwork initializes correctly for various parameters."""
    aedrnn = AEDRNNNetwork(**params)
    encoder, decoder = aedrnn.build_network((1000, 5))
    assert encoder is not None
    assert decoder is not None
