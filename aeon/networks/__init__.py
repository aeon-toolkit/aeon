"""Deep learning networks."""

__all__ = [
    "BaseDeepNetwork",
    "CNNNetwork",
    "EncoderNetwork",
    "FCNNetwork",
    "InceptionNetwork",
    "MLPNetwork",
    "ResNetNetwork",
    "TapNetNetwork",
    "AEFCNNetwork",
    "AEResNetNetwork",
    "LITENetwork",
]
from aeon.networks._ae_fcn import AEFCNNetwork
from aeon.networks._ae_resnet import AEResNetNetwork
from aeon.networks._cnn import CNNNetwork
from aeon.networks._encoder import EncoderNetwork
from aeon.networks._fcn import FCNNetwork
from aeon.networks._inception import InceptionNetwork
from aeon.networks._lite import LITENetwork
from aeon.networks._mlp import MLPNetwork
from aeon.networks._resnet import ResNetNetwork
from aeon.networks._tapnet import TapNetNetwork
from aeon.networks.base import BaseDeepNetwork
