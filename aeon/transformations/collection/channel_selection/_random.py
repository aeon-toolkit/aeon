"""Random channel selection."""

import math
import random

from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

__maintainer__ = ["TonyBagnall"]


class RandomChannelSelector(BaseChannelSelector):
    """Selects a random proportion of channels.

    Parameters
    ----------
    p: float, default 0.4
        proportion of channels to keep. If p*len(X) is non integer it is rounded up
        to the nearest integer.

    Attributes
    ----------
    channels_selected_ : list[int]
        List of channels selected in fit.
    """

    _tags = {
        "capability:multivariate": True,
        "requires_y": False,
    }

    def __init__(self, p=0.4):
        self.p = p
        super().__init__()

    def _fit(self, X, y):
        """Randomly select channels to retain."""
        to_select = math.ceil(self.p * len(X))
        self.channels_selected_ = random.sample(X[0].shape[1], to_select)
        return self
