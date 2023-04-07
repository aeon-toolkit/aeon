# -*- coding: utf-8 -*-
"""Implements base class for defining performance metric in aeon."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["rnkuhns", "fkiraly"]
__all__ = ["BaseMetric"]

from aeon.base import BaseObject


class BaseMetric(BaseObject):
    """Base class for defining metrics in aeon.

    Extends aeon BaseObject.
    """

    def __init__(self):
        super(BaseMetric, self).__init__()

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
