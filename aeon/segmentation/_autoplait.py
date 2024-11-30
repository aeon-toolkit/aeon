"""AutoPlait Segmentation."""

__maintainer__ = []
__all__ = ["AutoPlaitSegmenter"]

import numpy as np

from aeon.segmentation import BaseSegmenter


class AutoPlaitSegmenter(BaseSegmenter):
    """
        DOCSTRING
        """

    _tags = {}

    def __init__(self):
        super().__init__(axis=0)

    def _predict(self, X) -> np.ndarray:
        return