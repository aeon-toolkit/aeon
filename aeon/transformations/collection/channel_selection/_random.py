"""Random channel selection."""

import math

from sklearn.utils import check_random_state

from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

__maintainer__ = ["TonyBagnall"]
__all__ = ["RandomChannelSelector"]


class RandomChannelSelector(BaseChannelSelector):
    """Selects a random proportion of channels.

    Parameters
    ----------
    p: float, default 0.4
        proportion of channels to keep. If p*n_channels is non integer it is rounded up
        to the nearest integer.

    Attributes
    ----------
    channels_selected_ : list[int]
        List of channels selected in fit.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection.channel_selection import RandomChannelSelector # noqa
    >>> X = np.random.rand(10, 10, 100)
    >>> selector = RandomChannelSelector(p=0.4)
    >>> XNew = selector.fit_transform(X)
    >>> XNew.shape
    (10, 4, 100)
    """

    _tags = {
        "capability:multivariate": True,
        "requires_y": False,
    }

    def __init__(self, p=0.4, random_state=None):
        if p <= 0 or p > 1:
            raise ValueError(
                "Proportion of channels to select should be in the range (0,1]."
            )
        self.p = p
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        """Randomly select channels to retain."""
        rng = check_random_state(self.random_state)
        to_select = math.ceil(self.p * X.shape[1])
        self.channels_selected_ = rng.choice(
            list(range(X[0].shape[0])), size=to_select, replace=False
        )
        return self
