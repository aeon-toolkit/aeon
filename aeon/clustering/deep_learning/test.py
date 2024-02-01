"""Test suite for ae_resnet module.

This script contains unit tests and/or integration tests to validate the functionality,
performance, and behavior of the 'ae_resnet' Autoencoder with ResNet architecture."""

import sys

from aeon.utils.estimator_checks import check_estimator
from ae_resnet import ResNetClusterer

print(check_estimator(ResNetClusterer))
