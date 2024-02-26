"""Implements base class for defining performance metric in aeon."""

__maintainer__ = []
__all__ = ["BaseMetric"]

from aeon.base import BaseObject


class BaseMetric(BaseObject):
    """Base class for defining metrics in aeon.

    Extends aeon BaseObject.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : ground truth prediction target values
            type depending on the loss type, abstract descendants

        y_pred : predicted values
            type depending on the loss type, abstract descendants

        Returns
        -------
        loss : type depending on the loss type, abstract descendants
        """
        return self.evaluate(y_true, y_pred, **kwargs)

    def evaluate(self, y_true, y_pred, **kwargs):
        """Calculate metric value using underlying metric function.

        Parameters
        ----------
        y_true : ground truth prediction target values
            type depending on the loss type, abstract descendants

        y_pred : predicted values
            type depending on the loss type, abstract descendants

        Returns
        -------
        loss : float
            Calculated loss metric.
        """
        raise NotImplementedError("abstract method")
